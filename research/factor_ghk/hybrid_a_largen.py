"""Hybrid A at large n: the regime winning exists for.

    PYTHONPATH=. python3 research/factor_ghk/hybrid_a_largen.py --n 1000 --rank 3 --D 0.05 --R 32 128 512 2048 --workers 24

Same model and same estimators as hybrid_a.py, but n is 1e3 to 1e5, so:

  * every winner cannot be priced by the per-winner hybrid (n passes per
    draw); instead a set of TARGET winners is chosen from a cheap first
    estimate (the favourites plus a few mid-field runners), and the
    per-winner hybrid prices those.  Cost: R passes per target.
  * the fixed-node comparators price ALL n winners in one pass per node;
    they are scored on the targets (and, for the record, on all n).
  * the shared/mixture variant needs n chain masses per draw and is out.
  * GHK on the runner contrasts is O(n^2) per draw per winner plus an
    O(n^3) Cholesky; it is run on the targets only where that is
    affordable (n <= --ghk-max-n) and timed, as the accuracy spot-check.
  * the truth is a 2^m scrambled-Sobol rule with its own seed, computed
    in parallel over node chunks (the un-normalised chunk sums add), and
    certified against a second scramble and against MC.

Resumable: the truth streams, the hybrid per target, and GHK per target
are chunked over draw indices of fixed Sobol streams with running sums in
--state-dir; rerun with larger --R / --truth-m and only new draws are
priced.  CSV appended and flushed per row.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                                             # one thread per worker

import numpy as np                                                    # noqa: E402
from scipy.special import ndtri                                       # noqa: E402
from scipy.stats import norm                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_a as H                                                  # noqa: E402
from winning.methods.native import _ghk_prob                          # noqa: E402

G = {}                                                                # field shared with forked workers


# ------------------------------------------------------------ projections
def projection_interval(A, b, fixed, k, lo=-8.0, hi=8.0):
    """As hybrid_a.projection_interval, but the LAST free coordinate is a
    1-D intersection of half-lines and needs no LP (O(n), vectorised).
    Checked against the LP to 1e-9 in the smoke test."""
    r = A.shape[1]
    if k == r - 1:
        rhs = b - (A[:, :k] @ np.asarray(fixed, dtype=float) if k else 0.0)
        a = A[:, k]
        pos, neg, zero = a > 0, a < 0, a == 0
        if np.any(rhs[zero] < 0):
            return None
        ub = min(hi, float(np.min(rhs[pos] / a[pos])) if pos.any() else hi)
        lb = max(lo, float(np.max(rhs[neg] / a[neg])) if neg.any() else lo)
        return (lb, ub) if lb < ub else None
    return H.projection_interval(A, b, fixed, k, lo, hi)


# ---------------------------------------------------------------- workers
def _w_sobol_chunk(task):
    """Un-normalised sum of p(f) over Sobol nodes [lo, hi) of stream `seed`."""
    lo, hi, seed = task
    mu, V, D, points = G["mu"], G["V"], G["D"], G["points"]
    F = ndtri(H.sobol_slice(V.shape[1], seed, lo, hi - lo))
    return H.lattice_weighted(mu, V, D, F, np.ones(hi - lo), points)


def _w_mc_chunk(task):
    paths, seed = task
    mu, V, D = G["mu"], G["V"], G["D"]
    r = np.random.default_rng(seed)
    n, k = V.shape
    wins = np.zeros(n)
    done = 0
    step = max(1, min(paths, int(2e7 // n)))
    while done < paths:
        m = min(step, paths - done)
        X = mu + r.normal(size=(m, k)) @ V.T + np.sqrt(D) * r.normal(size=(m, n))
        wins += np.bincount(X.argmin(1), minlength=n)
        done += m
    return wins


def _w_hybrid_target(task):
    """Per-winner Hybrid A for target i over draws [done, done+extra).
    Returns (i, acc_i, passes, seconds)."""
    i, done, extra = task
    mu, V, D, points, margin = G["mu"], G["V"], G["D"], G["points"], G["margin"]
    t0 = time.perf_counter()
    n, k = V.shape
    U = H.sobol_slice(k, 7, done, extra)
    A, b = H.competitive_constraints(mu, V, D, i, margin)
    Fs, ws = [], []
    for row in range(extra):
        f = np.zeros(k); w = 1.0; ok = True
        for kk in range(k):
            iv = projection_interval(A, b, f[:kk], kk)
            if iv is None:
                ok = False; break
            f[kk], mass = H.truncated_draw(iv[0], iv[1], U[row, kk])
            w *= mass
        if ok and w > 0:
            Fs.append(f.copy()); ws.append(w)
    acc = float(H.lattice_weighted(mu, V, D, Fs, ws, points)[i]) if Fs else 0.0
    return i, acc, len(Fs), time.perf_counter() - t0


def _w_ghk_target(task):
    """qmc_ghk for target i over draws [done, done+extra): (i, sum of weights, seconds)."""
    i, done, extra = task
    mu, Sigma = G["mu"], G["Sigma"]
    n = len(mu)
    t0 = time.perf_counter()
    u = H.sobol_slice(n - 1, 13, done, extra)
    return i, _ghk_prob(-mu, Sigma, i, extra, u) * extra, time.perf_counter() - t0


# ------------------------------------------------------------------ state
def load(path, default):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


class Arrays:
    """Named float arrays in one .npz, rewritten atomically on change."""
    def __init__(self, path):
        self.path = path
        self.d = dict(np.load(path)) if os.path.exists(path) else {}
    def __contains__(self, k): return k in self.d
    def __getitem__(self, k): return self.d[k]
    def __setitem__(self, k, v):
        self.d[k] = np.asarray(v, dtype=float)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, **self.d); os.replace(tmp, self.path)


def sobol_stream(pool, arrays, state, seed, total, chunk):
    """Running un-normalised sum over the first `total` nodes of Sobol stream
    `seed`, extended in parallel from wherever it stands.  Returns
    (p, nodes actually used, seconds); if the state already holds more than
    `total` nodes the larger estimate is returned and reported as such."""
    key = f"sobol{seed}"; st = state.setdefault(key, {"done": 0, "seconds": 0.0})
    acc = arrays[key] if key in arrays else np.zeros(len(G["mu"]))
    if st["done"] < total:
        t = time.perf_counter()
        tasks = [(lo, min(lo + chunk, total), seed) for lo in range(st["done"], total, chunk)]
        for part in pool.imap_unordered(_w_sobol_chunk, tasks):
            acc = acc + part
        st["seconds"] += time.perf_counter() - t; st["done"] = total
        arrays[key] = acc
    return acc / st["done"], st["done"], st["seconds"]


def sobol_fixed(pool, arrays, state, m, workers):
    """The shipped rule's 2^m nodes (seed 0), all winners; cached per m."""
    key = f"cmp_m{m}"
    if key not in arrays:
        t = time.perf_counter(); total = 2 ** m; chunk = max(8, total // workers)
        acc = sum(pool.imap_unordered(_w_sobol_chunk, [(lo, min(lo + chunk, total), 0) for lo in range(0, total, chunk)]))
        arrays[key] = acc / total; state[key + "_seconds"] = time.perf_counter() - t
    return arrays[key], state[key + "_seconds"]


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--D", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--R", type=int, nargs="+", default=[32, 128, 512], help="hybrid draws per target; ascending")
    ap.add_argument("--sobol-m", type=int, nargs="+", default=[7, 9, 11, 13], help="fixed Sobol comparators 2^m (seed 0, the shipped stream)")
    ap.add_argument("--truth-m", type=int, default=16, help="truth = Sobol 2^m, seed 101; certified vs seed 202 and MC")
    ap.add_argument("--paths", type=int, default=1_000_000, help="MC paths for the certification")
    ap.add_argument("--top", type=int, default=8, help="favourites among the targets")
    ap.add_argument("--mid", type=float, nargs="*", default=[1e-2, 3e-3, 1e-3, 3e-4], help="mid-field targets: runner nearest each p")
    ap.add_argument("--ghk", type=int, nargs="+", default=[256, 1024], help="GHK draws per target; ascending")
    ap.add_argument("--ghk-max-n", type=int, default=2000, help="skip GHK above this n (O(n^2) per draw, O(n^3) Cholesky, n^2 memory)")
    ap.add_argument("--ghk-targets", type=int, default=None, help="run GHK on only the first k targets (memory: each worker holds ~3 n^2 doubles)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--state-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "state_largen"))
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    mu, V, D, sharp = H.make_field(args.n, args.rank, args.D, args.seed)
    G.update(mu=mu, V=V, D=D, points=args.points, margin=args.margin)
    key = f"n{args.n}_rank{args.rank}_D{args.D}_seed{args.seed}_margin{args.margin}_pts{args.points}"
    os.makedirs(args.state_dir, exist_ok=True)
    spath = os.path.join(args.state_dir, key + ".json"); apath = os.path.join(args.state_dir, key + ".npz")
    state = load(spath, {}); arrays = Arrays(apath)
    if args.n <= args.ghk_max_n:
        t = time.perf_counter(); G["Sigma"] = V @ V.T + np.diag(D); G["tSigma"] = time.perf_counter() - t
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers)
    chunk = max(8, 2 ** args.truth_m // (4 * args.workers))

    csv_w = None
    if args.csv:
        import csv
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        fh = open(args.csv, "a", newline="")
        csv_w = csv.DictWriter(fh, fieldnames=["n", "rank", "D", "seed", "margin", "sharp", "target", "truth_p", "method", "budget", "passes", "p", "abs_err", "rel_err", "seconds", "truth_m", "truth_vs_scramble", "truth_vs_mc"])
        if new: csv_w.writeheader(); fh.flush()

    print(f"\n=== n={args.n}, rank {args.rank}, D={args.D}, margin {args.margin}: sharpness {sharp:.1f}; {args.workers} workers; state {spath} ===", flush=True)

    # truth: two independent scrambles, extended in parallel
    total = 2 ** args.truth_m
    truth, used, t1 = sobol_stream(pool, arrays, state, 101, total, chunk); save(spath, state)
    check, _, t2 = sobol_stream(pool, arrays, state, 202, total, chunk); save(spath, state)
    args.truth_m = int(round(np.log2(used)))                          # the state may hold more than asked
    d_scr = float(np.abs(truth - check).max())
    mk = f"mc{args.paths}"
    if mk not in arrays:
        t = time.perf_counter()
        per = max(1, args.paths // args.workers)
        wins = sum(pool.imap_unordered(_w_mc_chunk, [(per, 1000 + w) for w in range(args.workers)]))
        arrays[mk] = wins / wins.sum(); state[mk + "_seconds"] = time.perf_counter() - t; save(spath, state)
    pmc = arrays[mk]; d_mc = float(np.abs(truth - pmc).max()); se = float(np.sqrt(truth.max() * (1 - truth.max()) / args.paths))
    print(f"    truth = Sobol 2^{args.truth_m} seed 101 ({t1:.0f}s); vs second scramble {d_scr:.1e}; vs MC {args.paths} paths {d_mc:.1e} (se ~{se:.1e}); read nothing below ~{2*d_scr:.0e}")
    print(f"    truth: top p {np.array2string(np.sort(truth)[::-1][:6], precision=4, floatmode='fixed')}; #p>1e-3 {(truth>1e-3).sum()}, #p>1e-4 {(truth>1e-4).sum()}")

    # targets: favourites + mid-field, chosen on the truth (the cheap first
    # estimate would pick the same runners; the truth is already paid for)
    order = np.argsort(-truth)
    targets = list(order[:args.top])
    for q in args.mid:
        j = int(np.argmin(np.abs(truth - q)))
        if j not in targets: targets.append(j)
    targets = [int(t) for t in targets]
    print(f"    targets ({len(targets)}): " + ", ".join(f"{i}:{truth[i]:.2e}" for i in targets), flush=True)

    rows = []
    def report(method, budget, passes, p_targets, dt, p_all=None):
        err = np.abs(p_targets - truth[targets]); rel = err / truth[targets]
        line = f"  {method:<34} {budget:>7} {passes:>7} {err.max():>10.1e} {rel.max():>9.1e}"
        if p_all is not None:
            line += f"   all-n max abs {np.abs(p_all - truth).max():.1e}"
        line += f" {dt:>8.1f}s"
        print(line, flush=True)
        if csv_w:
            for i, pt in zip(targets, p_targets):
                csv_w.writerow(dict(n=args.n, rank=args.rank, D=args.D, seed=args.seed, margin=args.margin, sharp=sharp, target=i, truth_p=truth[i],
                                    method=method, budget=budget, passes=passes, p=pt, abs_err=abs(pt - truth[i]), rel_err=abs(pt - truth[i]) / truth[i],
                                    seconds=dt, truth_m=args.truth_m, truth_vs_scramble=d_scr, truth_vs_mc=d_mc))
            fh.flush()

    print(f"  {'method':<34} {'budget':>7} {'passes':>7} {'max abs':>10} {'max rel':>9}   (over targets)", flush=True)
    # fixed Sobol comparators: the shipped stream (seed 0), all n winners per pass
    for m in args.sobol_m:
        p, dt = sobol_fixed(pool, arrays, state, m, args.workers); save(spath, state)
        report(f"fixed scrambled Sobol 2^{m}", 2 ** m, 2 ** m, p[targets], dt, p_all=p)
    # per-winner Hybrid A on the targets, one process per target, resumable
    hy = state.setdefault("hybrid", {})
    for R in args.R:
        tasks = []
        for i in targets:
            st = hy.setdefault(str(i), {"done": 0, "acc": 0.0, "passes": 0, "seconds": 0.0})
            if st["done"] < R:
                tasks.append((i, st["done"], R - st["done"]))
        for i, acc, passes, sec in pool.imap_unordered(_w_hybrid_target, tasks):
            st = hy[str(i)]; st["acc"] += acc; st["passes"] += passes; st["seconds"] += sec; st["done"] = R
        save(spath, state)
        if any(hy[str(i)]["done"] != R for i in targets):
            print(f"  Hybrid A R={R}: skipped, state already holds more draws for some targets"); continue
        p = np.array([hy[str(i)]["acc"] / R for i in targets])
        passes = sum(hy[str(i)]["passes"] for i in targets)
        secs = max(hy[str(i)]["seconds"] for i in targets)          # wall time with one process per target
        report(f"Hybrid A per-winner R={R} x {len(targets)} targets", R, passes, p, secs)
        print(f"      raw/normalised n/a at large n; hybrid passes {passes} = {passes/len(targets):.0f} per target; Sobol at 2^{int(np.log2(max(passes,1)))} passes prices all {args.n}")
    # GHK spot-check on the targets
    if args.n <= args.ghk_max_n:
        gk = state.setdefault("ghk", {})
        gt = targets[:args.ghk_targets] if args.ghk_targets else targets
        for B in args.ghk:
            tasks = []
            for i in gt:
                st = gk.setdefault(str(i), {"done": 0, "acc": 0.0, "seconds": 0.0})
                if st["done"] < B:
                    tasks.append((i, st["done"], B - st["done"]))
            for i, acc, sec in pool.imap_unordered(_w_ghk_target, tasks):
                st = gk[str(i)]; st["acc"] += acc; st["seconds"] += sec; st["done"] = B
            save(spath, state)
            if any(gk[str(i)]["done"] != B for i in gt):
                print(f"  qmc_ghk B={B}: skipped, state already holds more draws for some targets"); continue
            p = np.array([gk[str(i)]["acc"] / B for i in gt])
            err = np.abs(p - truth[gt]); rel = err / truth[gt]
            secs = sum(gk[str(i)]["seconds"] for i in gt)
            print(f"  {f'qmc_ghk (contrasts) B={B} x {len(gt)} targets':<34} {B:>7} {0:>7} {err.max():>10.1e} {rel.max():>9.1e} {secs:>8.1f}s  ({secs/len(gt):.1f}s per target, single thread; Sigma build {G['tSigma']:.1f}s)", flush=True)
            if csv_w:
                for i, pt in zip(gt, p):
                    csv_w.writerow(dict(n=args.n, rank=args.rank, D=args.D, seed=args.seed, margin=args.margin, sharp=sharp, target=i, truth_p=truth[i],
                                        method=f"qmc_ghk (contrasts) B={B}", budget=B, passes=0, p=pt, abs_err=abs(pt - truth[i]), rel_err=abs(pt - truth[i]) / truth[i],
                                        seconds=secs, truth_m=args.truth_m, truth_vs_scramble=d_scr, truth_vs_mc=d_mc))
                fh.flush()
        print(f"      all {args.n} winners by GHK would cost {args.n/len(gt):.0f}x the per-target time above")
    else:
        print(f"  qmc_ghk skipped: n={args.n} > --ghk-max-n {args.ghk_max_n} (Sigma {8*args.n**2/1e9:.1f} GB, O(n^3) Cholesky per winner)")
    pool.close(); pool.join()
    if csv_w:
        fh.close(); print(f"\nappended to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
