"""Node rules B: improvements to the f-quadrature that are SHARED by all
runners (one lattice pass per node prices everyone), tested against the
certified truths that hybrid_a.py / hybrid_a_largen.py left in runs/state*.

    PYTHONPATH=. python3 research/factor_ghk/nodes_b.py --exp small  --csv runs/nodes_b_small.csv
    PYTHONPATH=. python3 research/factor_ghk/nodes_b.py --exp largen --n 1000 --csv runs/nodes_b_n1000.csv
    PYTHONPATH=. python3 research/factor_ghk/nodes_b.py --exp prune  --n 100000 --csv runs/nodes_b_prune.csv

1. rotate   f ~ N(0, I) is rotation invariant, so X = mu + V f = mu + (V Q)(Q'f).
            Choose Q so Sobol coordinate 1 (the best distributed) lies along
            the direction in which the conditional race is sharpest: the top
            right-singular vector of the centred loadings, optionally weighted
            by a pilot estimate of who can win.  'reverse' puts it last, as a
            control.  Zero runtime cost if it works.
2. adaptive Stratified Sobol on the unit cube of f-quantiles.  Start with one
            cell and k points; repeatedly split the cell with the largest
            volume x variation, where variation = sum_i (max - min) of p_i(f)
            over the cell's points, i.e. how much winner mass changes hands
            inside the cell; each child gets k fresh points.  Estimate =
            sum over leaves of volume x mean p.  Draws go where p varies, for
            every runner at once, no LPs.
3. prune    Large n only.  From a 2^7 pilot, keep the runners whose conditional
            mean comes within `margin` idiosyncratic sd of the leader at some
            pilot node; price the pruned field with the same Sobol rule.  A
            runner that never comes close contributes a survival factor of ~1
            to everyone else, so dropping it moves nobody by more than its own
            chance.  Buys pass count, not accuracy per pass; scored on all n.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np                                                    # noqa: E402
from scipy.special import ndtri                                       # noqa: E402
from scipy.stats import qmc                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_a as H                                                  # noqa: E402
import winning.factor as wf                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ truths
def small_truth(rank, D):
    path = os.path.join(HERE, "runs", "state", f"rank{rank}_n8_D{D}_seed3_margin4.0_pts257.json")
    st = json.load(open(path))
    key = max((k for k in st if k.startswith("truth_m")), key=lambda k: int(k.split("_")[1][1:]))
    return np.asarray(st[key]["p"]), st[key]["d_scr"], key


def largen_truth(n):
    base = os.path.join(HERE, "runs", "state_largen", f"n{n}_rank3_D0.05_seed3_margin4.0_pts257")
    st = json.load(open(base + ".json")); arr = np.load(base + ".npz")
    p = arr["sobol101"] / st["sobol101"]["done"]; q = arr["sobol202"] / st["sobol202"]["done"]
    return p, float(np.abs(p - q).max()), f"sobol 2^{int(np.log2(st['sobol101']['done']))}"


# -------------------------------------------------------------- estimators
def sobol_nodes(r, m, seed):
    return ndtri(qmc.Sobol(r, scramble=True, seed=seed).random_base2(m))


def price(mu, V, D, F, points=257):
    return np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F, W=np.full(len(F), 1.0 / len(F)), points=points), dtype=float)


def rotation(V, weights=None, reverse=False):
    """Q (r x r orthogonal) whose first column is the sharp direction."""
    w = np.ones(len(V)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    Vc = V - (w[:, None] * V).sum(0)
    _, s, Wt = np.linalg.svd(np.sqrt(w)[:, None] * Vc, full_matrices=False)
    Q = Wt.T                                                            # columns = right singular vectors, descending
    if Q.shape[1] < V.shape[1]:                                        # rank-deficient weights: complete the basis
        Q, _ = np.linalg.qr(np.hstack([Q, np.random.default_rng(0).normal(size=(V.shape[1], V.shape[1] - Q.shape[1]))]))
    return Q[:, ::-1] if reverse else Q


def adaptive_tree(mu, V, D, budget, k=8, seed=0, points=257):
    """Adaptive stratified Sobol; returns (p, passes, leaves)."""
    r = V.shape[1]
    eng = qmc.Sobol(r, scramble=True, seed=seed)
    def evaluate(U):
        return price(mu, V, D, ndtri(np.clip(U, 1e-12, 1 - 1e-12)), points) * 1.0  # normalised = mean over the k points
    def new_cell(lo, hi, U_in, P_in):
        Un = lo + (hi - lo) * eng.random(k)
        Pn = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=ndtri(np.clip(Un, 1e-12, 1 - 1e-12)), W=np.full(k, 1.0 / k), points=points), dtype=float)
        # need per-point p for the variation: one call per point is too slow; use the k-point mean for the
        # estimate and finite differences between the child's mean and the parent's for the variation proxy
        return dict(lo=lo, hi=hi, mean=Pn if not len(U_in) else (Pn * k + P_in.sum(0)) / (k + len(U_in)),
                    U=np.vstack([U_in, Un]) if len(U_in) else Un, P=np.vstack([P_in, np.tile(Pn, (k, 1))]) if len(U_in) else np.tile(Pn, (k, 1)))
    # per-point p is needed for a variation measure; price the k points individually but batched over cells is
    # awkward, so accept k calls per cell (30 us each at n=8; 4 ms at n=1000).
    def per_point(U):
        return np.array([np.asarray(wf.race_probabilities(mu + V @ ndtri(np.clip(u, 1e-12, 1 - 1e-12)), D=D, points=points), dtype=float) for u in U])
    passes = 0
    U0 = eng.random(k); P0 = per_point(U0); passes += k
    cells = [dict(lo=np.zeros(r), hi=np.ones(r), U=U0, P=P0)]
    def score(c):
        vol = float(np.prod(c["hi"] - c["lo"]))
        var = float((c["P"].max(0) - c["P"].min(0)).sum()) if len(c["P"]) > 1 else 2.0
        return vol * var
    while passes + 2 * k <= budget:
        j = int(np.argmax([score(c) for c in cells])); c = cells.pop(j)
        d = int(np.argmax(c["hi"] - c["lo"])); mid = 0.5 * (c["lo"][d] + c["hi"][d])
        for side in (0, 1):
            lo, hi = c["lo"].copy(), c["hi"].copy()
            if side == 0: hi[d] = mid
            else: lo[d] = mid
            inside = (c["U"][:, d] < mid) if side == 0 else (c["U"][:, d] >= mid)
            Un = lo + (hi - lo) * eng.random(k); Pn = per_point(Un); passes += k
            cells.append(dict(lo=lo, hi=hi, U=np.vstack([c["U"][inside], Un]), P=np.vstack([c["P"][inside], Pn])))
    p = np.zeros(len(mu))
    for c in cells:
        p += float(np.prod(c["hi"] - c["lo"])) * c["P"].mean(0)
    return p, passes, len(cells)


def prune_field(mu, V, D, F_pilot, margin):
    M = mu[None, :] + F_pilot @ V.T                                     # (nodes, n) conditional means
    lead = M.min(1, keepdims=True)
    slack = ((M - lead) / np.sqrt(2.0 * D)[None, :]).min(0)             # best case over pilot nodes, in sd units
    return slack < margin


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", choices=["small", "largen", "prune"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--D", type=float, nargs="+", default=[0.1, 0.05, 0.02])
    ap.add_argument("--m", type=int, nargs="+", default=[9, 11, 13], help="budgets 2^m passes")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="scramble seeds; seed 0 is the shipped stream")
    ap.add_argument("--k", type=int, default=8, help="adaptive: points per cell")
    ap.add_argument("--margin", type=float, default=5.0, help="prune: keep runners within this many sd of the leader at some pilot node")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    rows = []
    def emit(**kw):
        rows.append(kw)
        if args.csv:
            new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
            with open(args.csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(kw)); 
                if new: w.writeheader()
                w.writerow(kw)

    def run_field(label, mu, V, D, truth, d_scr, targets=None):
        n, r = V.shape
        print(f"\n=== {label}: truth certified to {d_scr:.1e} ===")
        print(f"  {'method':<28} {'passes':>6} " + " ".join(f"{'seed'+str(s):>9}" for s in args.seeds) + f" {'mean':>9} {'max':>9}" + ("   targets max rel (seed mean)" if targets is not None else ""), flush=True)
        pilot = price(mu, V, D, sobol_nodes(r, 7, 0))
        rots = {"rotate svd": rotation(V), "rotate svd p-weighted": rotation(V, weights=np.maximum(pilot, 1e-9)), "rotate reverse (control)": rotation(V, reverse=True)}
        for m in args.m:
            B = 2 ** m
            def report(method, ps, passes, secs):
                errs = np.array([np.abs(p - truth).max() for p in ps])
                line = f"  {method:<28} {passes:>6} " + " ".join(f"{e:>9.1e}" for e in errs) + f" {errs.mean():>9.1e} {errs.max():>9.1e}"
                rel = None
                if targets is not None:
                    rel = float(np.mean([np.max(np.abs(p[targets] - truth[targets]) / truth[targets]) for p in ps])); line += f"   {rel:.1e}"
                print(line + f"  {secs:.1f}s", flush=True)
                emit(field=label, method=method, passes=passes, err_seed0=errs[0], err_mean=errs.mean(), err_max=errs.max(), targets_max_rel=rel, seconds=secs)
            t = time.perf_counter(); ps = [price(mu, V, D, sobol_nodes(r, m, s)) for s in args.seeds]; report(f"Sobol 2^{m}", ps, B, time.perf_counter() - t)
            for name, Q in rots.items():
                t = time.perf_counter(); ps = [price(mu, V @ Q, D, sobol_nodes(r, m, s)) for s in args.seeds]; report(f"{name} 2^{m}", ps, B, time.perf_counter() - t)
            t = time.perf_counter(); out = [adaptive_tree(mu, V, D, B, k=args.k, seed=s) for s in args.seeds]
            report(f"adaptive tree k={args.k}", [o[0] for o in out], int(np.mean([o[1] for o in out])), time.perf_counter() - t)

    if args.exp == "small":
        for rank in args.ranks:
            for D_level in args.D:
                mu, V, D, sharp = H.make_field(8, rank, D_level, 3)
                truth, d_scr, key = small_truth(rank, D_level)
                run_field(f"n=8 rank {rank} D {D_level} (sharpness {sharp:.0f})", mu, V, D, truth, d_scr)
    elif args.exp == "largen":
        mu, V, D, sharp = H.make_field(args.n, 3, 0.05, 3)
        truth, d_scr, key = largen_truth(args.n)
        order = np.argsort(-truth); targets = list(order[:8]) + [int(np.argmin(np.abs(truth - q))) for q in (1e-2, 3e-3, 1e-3, 3e-4)]
        run_field(f"n={args.n} rank 3 D 0.05 (sharpness {sharp:.0f})", mu, V, D, truth, d_scr, targets=targets)
    else:
        mu, V, D, sharp = H.make_field(args.n, 3, 0.05, 3)
        truth, d_scr, key = largen_truth(args.n)
        order = np.argsort(-truth); targets = list(order[:8]) + [int(np.argmin(np.abs(truth - q))) for q in (1e-2, 3e-3, 1e-3, 3e-4)]
        print(f"\n=== prune n={args.n} rank 3 D 0.05: truth certified to {d_scr:.1e}; #truth>1e-4 {(truth>1e-4).sum()}, >1e-6 {(truth>1e-6).sum()} ===")
        t = time.perf_counter(); F0 = sobol_nodes(3, 7, 0); M_pilot_p = price(mu, V, D, F0); t_pilot = time.perf_counter() - t
        for margin in (3.0, args.margin, 8.0):
            keep = prune_field(mu, V, D, F0, margin); nk = int(keep.sum())
            dropped_mass = float(truth[~keep].sum()); worst_dropped = float(truth[~keep].max()) if (~keep).any() else 0.0
            print(f"  margin {margin}: keep {nk} of {args.n}; truth mass dropped {dropped_mass:.1e}, largest dropped runner {worst_dropped:.1e}")
            for m in args.m:
                t = time.perf_counter(); p = np.zeros(args.n); p[keep] = price(mu[keep], V[keep], D[keep], sobol_nodes(3, m, 0)); dt = time.perf_counter() - t
                err = float(np.abs(p - truth).max()); rel = float(np.max(np.abs(p[targets] - truth[targets]) / truth[targets]))
                print(f"    pruned Sobol 2^{m}: all-n max abs {err:.1e}, targets max rel {rel:.1e}, {dt:.1f}s (+ pilot {t_pilot:.1f}s)", flush=True)
                emit(field=f"n={args.n}", method=f"pruned margin {margin} Sobol 2^{m}", keep=nk, passes=2 ** m, err_max=err, targets_max_rel=rel, dropped_mass=dropped_mass, seconds=dt, pilot_seconds=t_pilot)
        print(f"  unpruned pilot Sobol 2^7 for reference: all-n max abs {np.abs(M_pilot_p - truth).max():.1e}, {t_pilot:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
