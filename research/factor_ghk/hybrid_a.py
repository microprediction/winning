"""Hybrid A: GHK's sequential truncated importance sampling over the FACTOR
dimensions, with winning's lattice pricing the conditional race exactly at
each draw.  Compared against fixed factor nodes at equal lattice passes.

    python research/factor_ghk/hybrid_a.py --rank 3 --n 8 --D 0.05 --paths 3000000
    python research/factor_ghk/hybrid_a.py --rank 1 --n 8 --D 0.05 0.01

Model: X = mu + V f + sqrt(D) eps, f ~ N(0, I_r), min wins.  The lattice
integrates out eps exactly given f (one pass, all runners); the question is
how to integrate over f.

  fixed nodes      Gauss-Hermite tensor / scrambled Sobol / midpoint grid:
                   the shipped rules.  One lattice pass per node, ALL winners.
  Hybrid A         for each winner i, draw f sequentially: f_1 from N(0,1)
                   truncated to the exact projection of the polytope
                   {f : i is competitive} onto f_1; f_2 from N(0,1) truncated
                   to its projection given f_1; ...  GHK weight = product of
                   truncation masses.  p_i(f) from one lattice pass per draw.
                   Cost: n * R passes (per-winner draws).
  Hybrid A shared  one set of draws f ~ mixture over winners' truncated
                   proposals; one lattice pass per draw gives p_j(f) for ALL
                   j; each j is weighted by phi(f) / q_mix(f).  Cost: R passes.

"Competitive" for winner i at margin m (in idiosyncratic sd units):
    mu_i + V_i.f <= mu_j + V_j.f + m * sqrt(D_i + D_j)   for all j != i,
a polytope in f.  Its exact projection onto f_k given f_<k is found by two
small LPs (min / max f_k over the remaining dims), which is what makes this
GHK rather than a box heuristic.  At rank 1 the projection is closed form.

Success criterion (see README.md): Hybrid A error <= Sobol error at equal
total lattice passes, on fields where the shipped rule is weakest (sharp,
rank 2-5).  Report BOTH raw and normalised hybrid estimates; a raw sum far
from 1 means the margin is leaking mass.

Also reported: the shipped qmc_ghk on the N-1 runner contrasts (the
accuracy baseline; no lattice), and a truth from a 2^m scrambled-Sobol rule
with its own seed, certified against a second scramble and against MC.

Resumable: running sums per method live in --state-dir; rerun with larger
--R / --ghk / --truth-m and only the new draws are priced.  Outcome
(2026-09-22): the hybrid loses at every rank 1-5; see RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np                                                    # noqa: E402
from scipy.optimize import linprog                                    # noqa: E402
from scipy.special import ndtri                                       # noqa: E402
from scipy.stats import norm, qmc                                     # noqa: E402

import winning.factor as wf                                           # noqa: E402
from winning.factor.core import hermite_nodes, qmc_nodes              # noqa: E402
from winning.factor.races import _setup as _races_setup                # noqa: E402
from winning.methods.native import _ghk_prob                          # noqa: E402


# ------------------------------------------------------------------ field
def make_field(n, rank, D_level, seed):
    rng = np.random.default_rng(seed)
    mu = np.sort(rng.normal(size=n)) * 0.5
    V = rng.normal(size=(n, rank))
    D = np.full(n, float(D_level))
    Vc = V - V.mean(axis=0)
    sharp = float(np.sqrt(2.0) * np.max(np.sqrt((Vc ** 2).sum(1)) / np.sqrt(D)))
    return mu, V, D, sharp


def mc_truth(mu, V, D, paths, seed=0):
    r = np.random.default_rng(seed)
    n, k = V.shape
    wins = np.zeros(n)
    done = 0
    while done < paths:
        m = min(400_000, paths - done)
        X = mu + r.normal(size=(m, k)) @ V.T + np.sqrt(D) * r.normal(size=(m, n))
        wins += np.bincount(X.argmin(1), minlength=n)
        done += m
    return wins / paths


def certified_truth(mu, V, D, points, paths, m=18):
    """Reference for the f-quadrature.  The lattice prices the conditional
    race to ~1e-15 at any resolution (measured: points 65..2001 agree to
    5e-8, 129..2001 to 1e-14), so the only error left is the quadrature
    over f, and a 2^m scrambled-Sobol rule with a seed the compared rules
    never use is a far better reference than MC at 3M paths (se ~3e-4).
    Certified two ways: against a second independent scramble, and against
    plain MC.  Returns (truth, |truth - second scramble|, |truth - MC|, mc_se)."""
    k = V.shape[1]
    F1, W1 = qmc_nodes(k, m=m, seed=101)
    F2, W2 = qmc_nodes(k, m=m, seed=202)
    p1 = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F1, W=W1, points=points), dtype=float)
    p2 = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F2, W=W2, points=points), dtype=float)
    pmc = mc_truth(mu, V, D, paths)
    se = float(np.sqrt(pmc.max() * (1 - pmc.max()) / paths))
    return p1, float(np.abs(p1 - p2).max()), float(np.abs(p1 - pmc).max()), se


def lattice_conditional(mu, V, D, f, points):
    """Exact conditional race at factor value f: one lattice pass, all runners."""
    return np.asarray(wf.race_probabilities(mu + V @ f, D=D, points=points), dtype=float)


def lattice_weighted(mu, V, D, Fs, ws, points):
    """sum_r ws[r] * p(f_r), all runners, in ONE batched engine call (one
    lattice pass per node inside).  The engine normalises its output to sum
    1 whatever the weights, and each conditional race sums to 1 exactly, so
    the un-normalised sum is p_norm * sum(ws).  Agrees with the per-draw loop
    to ~1e-12 (checked at rank 1 and 3)."""
    Fs = np.asarray(Fs, dtype=float); ws = np.asarray(ws, dtype=float)
    if len(ws) == 0:
        return np.zeros(len(mu))
    p = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=Fs, W=ws, points=points), dtype=float)
    return p * ws.sum()


# ------------------------------------------------------- competitive polytope
def competitive_constraints(mu, V, D, i, margin):
    """A f <= b  <=>  i competitive.  Rows over j != i."""
    n, k = V.shape
    A = np.empty((n - 1, k)); b = np.empty(n - 1)
    row = 0
    for j in range(n):
        if j == i:
            continue
        A[row] = V[i] - V[j]
        b[row] = (mu[j] - mu[i]) + margin * np.sqrt(D[i] + D[j])
        row += 1
    return A, b


def projection_interval(A, b, fixed, k, lo=-8.0, hi=8.0):
    """[min, max] of f_k over the polytope A f <= b with f_<k = fixed and
    f_>k free in [lo, hi].  Exact via two LPs; None if empty."""
    r = A.shape[1]
    free = list(range(k, r))
    if not free:
        return None
    # substitute the fixed coordinates
    b_eff = b - A[:, :k] @ np.asarray(fixed, dtype=float) if k else b.copy()
    A_free = A[:, k:]
    bounds = [(lo, hi)] * len(free)
    c = np.zeros(len(free)); c[0] = 1.0
    out = []
    for sign in (1.0, -1.0):
        res = linprog(sign * c, A_ub=A_free, b_ub=b_eff, bounds=bounds, method="highs")
        if res.status != 0:
            return None
        out.append(sign * res.fun)
    a, bb = min(out), max(out)
    return (a, bb) if a < bb else None


def truncated_draw(a, b, u):
    Fa, Fb = norm.cdf(a), norm.cdf(b)
    mass = Fb - Fa
    return ndtri(Fa + u * mass), mass


# ------------------------------------------------------------------ hybrids
# Every stochastic method below is written as a CHUNK over draw indices
# [done, done + extra) of a fixed scrambled-Sobol stream (same seed), so a
# run can be resumed to a larger budget: the state file keeps the running
# weighted sums and the number of draws consumed, and extending R only
# prices the new draws.  Scrambled Sobol with a fixed seed is prefix-
# consistent (fast_forward(done).random(extra) == random(done+extra)[done:]).

def sobol_slice(d, seed, done, extra):
    """Points [done, done+extra) of the scrambled Sobol stream (d, seed)."""
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    if done:
        eng.fast_forward(done)                      # scipy rejects fast_forward(0)
    return eng.random(extra)


def hybrid_a_chunk(mu, V, D, done, extra, points, margin, seed=7):
    """Per-winner GHK in factor space over draws [done, done+extra).
    Returns (acc, passes): acc[i] = sum_r w_r p_i(f_r) over the chunk."""
    n, k = V.shape
    U = sobol_slice(k, seed, done, extra)
    acc = np.zeros(n); passes = 0
    for i in range(n):
        A, b = competitive_constraints(mu, V, D, i, margin)
        Fs = []; ws = []
        for row in range(extra):
            f = np.zeros(k); w = 1.0; ok = True
            for kk in range(k):
                iv = projection_interval(A, b, f[:kk], kk)
                if iv is None:
                    ok = False; break
                f[kk], mass = truncated_draw(iv[0], iv[1], U[row, kk])
                w *= mass
            if not ok or w <= 0:
                continue
            Fs.append(f.copy()); ws.append(w)
        if Fs:
            acc[i] = lattice_weighted(mu, V, D, Fs, ws, points)[i]
        passes += len(Fs)
    return acc, passes


def chain_mass(A, b, f):
    """GHK chain mass of f under winner-polytope (A, b): the product of the
    truncation masses along f_1, f_2 | f_1, ...  Zero if f leaves the polytope."""
    k = len(f)
    if not np.all(A @ f <= b + 1e-12):
        return 0.0
    w = 1.0
    for kk in range(k):
        iv = projection_interval(A, b, f[:kk], kk)
        if iv is None:
            return 0.0
        w *= norm.cdf(iv[1]) - norm.cdf(iv[0])
    return w


def hybrid_a_shared_chunk(mu, V, D, done, extra, points, margin, seed=7):
    """Shared draws from a mixture of the winners' truncated proposals over
    draws [done, done+extra); one lattice pass per draw serves every winner.
    Returns (acc, passes): acc = sum_r p(f_r) phi(f_r)/q_mix(f_r)."""
    n, k = V.shape
    U = sobol_slice(k + 1, seed, done, extra)
    cons = [competitive_constraints(mu, V, D, i, margin) for i in range(n)]
    Fs = []; ws = []
    for row in range(extra):
        i = min(int(U[row, k] * n), n - 1)                            # mixture component
        A, b = cons[i]
        f = np.zeros(k); ok = True
        for kk in range(k):
            iv = projection_interval(A, b, f[:kk], kk)
            if iv is None:
                ok = False; break
            f[kk], _ = truncated_draw(iv[0], iv[1], U[row, kk])
        if not ok:
            continue
        # mixture density q(f) = (1/n) sum_j phi(f) [f in poly_j] / mass_j(f); the chain
        # masses depend on the path, so recompute each j's chain mass at f
        q_mix = 0.0
        for j in range(n):
            wj = chain_mass(cons[j][0], cons[j][1], f)
            if wj > 0:
                q_mix += 1.0 / wj
        q_mix /= n
        if q_mix <= 0:
            continue
        Fs.append(f.copy()); ws.append(1.0 / q_mix)
    return lattice_weighted(mu, V, D, Fs, ws, points), len(Fs)


def qmc_ghk_chunk(mu, V, D, done, extra, seed=13):
    """The shipped winning.methods.native.qmc_ghk (Genz-Bretz GHK on the N-1
    runner contrasts, scrambled-Sobol uniforms) over draws [done, done+extra).
    Returns acc: per-winner SUM of the GHK weights over the chunk."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    u = sobol_slice(n - 1, seed, done, extra)
    # _ghk_prob is max-wins (P(i has the max)); winning is min-wins, and
    # min X = max(-X) with the same covariance, so pass -mu.
    return np.array([_ghk_prob(-mu, Sigma, i, extra, u) for i in range(n)]) * extra


# ------------------------------------------------------------------- state
def field_key(args, D_level):
    return f"rank{args.rank}_n{args.n}_D{D_level}_seed{args.seed}_margin{args.margin}_pts{args.points}"


def load_state(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def extend(state, key, R, chunk_fn):
    """Bring the running sums under state[key] up to R draws, pricing only
    the draws not yet consumed.  Returns (acc, passes, cumulative_seconds)."""
    st = state.get(key, {"done": 0, "acc": None, "passes": 0, "seconds": 0.0})
    if st["done"] < R:
        t = time.perf_counter()
        acc, passes = chunk_fn(st["done"], R - st["done"])
        st["seconds"] += time.perf_counter() - t
        st["acc"] = (np.asarray(acc) + (np.asarray(st["acc"]) if st["acc"] is not None else 0.0)).tolist()
        st["passes"] += int(passes)
        st["done"] = R
        state[key] = st
    if st["done"] > R:
        raise SystemExit(f"{key}: state holds {st['done']} draws, more than the requested {R}; "
                         f"request a budget >= {st['done']} or delete the state file")
    return np.asarray(st["acc"]), st["passes"], st["seconds"]


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--D", type=float, nargs="+", default=[0.05])
    ap.add_argument("--R", type=int, nargs="+", default=[16, 64, 256], help="hybrid draws (per winner); shared uses n*R; ascending")
    ap.add_argument("--ghk", type=int, nargs="+", default=[128, 512, 2048, 8192], help="qmc_ghk draw budgets; ascending")
    ap.add_argument("--paths", type=int, default=2_000_000, help="MC paths used only to certify the truth")
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--margin", type=float, default=4.0, help="competitive margin, idiosyncratic sd units")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--truth-m", type=int, default=18, help="truth = scrambled Sobol 2^m, seed 101 (certified vs seed 202 and MC)")
    ap.add_argument("--state-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "state"),
                    help="where running sums live; rerun with larger --R/--ghk/--truth-m to resume")
    ap.add_argument("--csv", type=str, default=None, help="appended to, one row per line, flushed as it goes")
    args = ap.parse_args()
    os.makedirs(args.state_dir, exist_ok=True)
    csv_fh = None
    if args.csv:
        import csv
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_fh = open(args.csv, "a", newline="")
        fields = ["rank", "n", "D", "seed", "margin", "points", "sharp", "method", "budget", "passes", "err", "raw_sum", "seconds", "truth_m", "truth_vs_scramble", "truth_vs_mc"]
        csv_w = csv.DictWriter(csv_fh, fieldnames=fields)
        if new:
            csv_w.writeheader(); csv_fh.flush()
    for D_level in args.D:
        mu, V, D, sharp = make_field(args.n, args.rank, D_level, args.seed)
        spath = os.path.join(args.state_dir, field_key(args, D_level) + ".json")
        state = load_state(spath)
        tk = f"truth_m{args.truth_m}_paths{args.paths}"
        if tk not in state:
            t0 = time.perf_counter()
            truth, d_scr, d_mc, se = certified_truth(mu, V, D, args.points, args.paths, m=args.truth_m)
            state[tk] = dict(p=truth.tolist(), d_scr=d_scr, d_mc=d_mc, se=se, seconds=time.perf_counter() - t0)
            save_state(spath, state)
        T = state[tk]; truth = np.asarray(T["p"]); d_scr, d_mc, se = T["d_scr"], T["d_mc"], T["se"]
        print(f"\n=== rank {args.rank}, n={args.n}, D={D_level}, margin {args.margin}: sharpness {sharp:.1f}; truth = Sobol 2^{args.truth_m} (seed 101), {T['seconds']:.1f}s ===", flush=True)
        print(f"    truth vs second scramble {d_scr:.1e}; vs MC {args.paths} paths {d_mc:.1e} (MC se ~{se:.1e}); read nothing below ~{max(2*d_scr, 1e-6):.0e}")
        print(f"    truth p = {np.array2string(truth, precision=4, floatmode='fixed', max_line_width=200)}")
        print(f"    state: {spath}")
        print(f"  {'method':<40} {'budget':>7} {'passes':>7} {'max|p-truth|':>13} {'raw sum':>8} {'s':>7}", flush=True)

        def report(name, p, budget, passes, dt, raw_sum=None):
            err = float(np.abs(p - truth).max())
            print(f"  {name:<40} {budget:>7} {passes:>7} {err:>13.1e} {('' if raw_sum is None else f'{raw_sum:8.4f}'):>8} {dt:>7.1f}", flush=True)
            if csv_fh:
                csv_w.writerow(dict(rank=args.rank, n=args.n, D=D_level, seed=args.seed, margin=args.margin, points=args.points, sharp=sharp,
                                    method=name, budget=budget, passes=passes, err=err, raw_sum=raw_sum, seconds=dt,
                                    truth_m=args.truth_m, truth_vs_scramble=d_scr, truth_vs_mc=d_mc))
                csv_fh.flush()

        # fixed-node rules: cheap, recomputed every run
        F0, W0 = _races_setup(mu, V, D, None, None, "normal")[3:5]
        t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, points=args.points)); report("winning default rule", p, len(F0), len(F0), time.perf_counter() - t)
        for Q in (15, 41) if args.rank >= 2 else (15, 51, 201):
            F, W = hermite_nodes(args.rank, Q=Q)
            t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F, W=W, points=args.points)); report(f"fixed Gauss-Hermite Q={Q}", p, len(F), len(F), time.perf_counter() - t)
        for m in (7, 9, 11, 13):
            F, W = qmc_nodes(args.rank, m=m)
            t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F, W=W, points=args.points)); report(f"fixed scrambled Sobol 2^{m}", p, len(F), len(F), time.perf_counter() - t)
        # GHK on the runner contrasts: the accuracy baseline, no lattice passes
        for B in args.ghk:
            acc, _, dt = extend(state, "qmc_ghk", B, lambda done, extra: (qmc_ghk_chunk(mu, V, D, done, extra), 0))
            save_state(spath, state)
            p = acc / B
            report(f"qmc_ghk (runner contrasts) B={B} (raw)", p, B, 0, dt, p.sum()); report(f"qmc_ghk (runner contrasts) B={B} (normalised)", p / p.sum(), B, 0, dt)
        # the hybrids, resumable
        for R in args.R:
            acc, passes, dt = extend(state, "hybrid_a", R, lambda done, extra: hybrid_a_chunk(mu, V, D, done, extra, args.points, args.margin))
            save_state(spath, state)
            p = acc / R
            report(f"Hybrid A per-winner R={R} (raw)", p, R, passes, dt, p.sum()); report(f"Hybrid A per-winner R={R} (normalised)", p / p.sum(), R, passes, dt)
            Rs = R * args.n
            acc, passes, dt = extend(state, "hybrid_a_shared", Rs, lambda done, extra: hybrid_a_shared_chunk(mu, V, D, done, extra, args.points, args.margin))
            save_state(spath, state)
            p = acc / Rs
            report(f"Hybrid A shared R={Rs} (raw)", p, Rs, passes, dt, p.sum()); report(f"Hybrid A shared R={Rs} (normalised)", p / p.sum(), Rs, passes, dt)
    if csv_fh:
        csv_fh.close(); print(f"\nappended to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
