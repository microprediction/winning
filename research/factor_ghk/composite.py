"""Composite horses: one runner out of many poor ones, so a lattice pass
costs (kept + groups) x L instead of n x L at large n.

    PYTHONPATH=. python3 research/factor_ghk/composite.py --n 10000 --m 11 13 --groups 0 20 100 --csv runs/composite.csv

Field, truth and targets as in hybrid_a_largen.py (rank 3, D 0.05, seed 3;
truth from runs/state_largen).  Steps:

  1. pilot   Sobol 2^7 nodes; runner j is KEPT if its conditional mean comes
             within --margin sd of the leader at some pilot node (a poor
             runner matters only where its factor contrast with the field is
             near its maximum; the pilot looks for that f).
  2. group   the rest are clustered by loading vector: k-means on the rows
             of V ('kmeans'), or 1-D seriation along the top principal
             direction of their loadings cut into contiguous groups
             ('seriation').  groups=0 is pure pruning.
  3. composite  group g becomes ONE runner with loading V_g (mean loading)
             and idiosyncratic distribution Z_g = min_j (mu_j + s_j eps_j),
             s_j^2 = D_j + ||V_j - V_g||^2: the exact distribution of the
             group minimum given a common factor value, with each member's
             deviation from the group loading folded into its own noise
             (independent-noise approximation; exact when the group's
             loadings coincide).  log-survival and log-density tabulated once.
  4. price   a small numpy lattice (same kernel as winning.factor.core, with
             per-runner tabulated distributions) at Sobol 2^m nodes.

Scored against the certified truth on the KEPT runners (max abs, and max
relative on the 12 targets), and on each group's total against the truth's
group sum.  Compared with pruning (groups=0) and with the full field at the
same nodes (from the largen logs; recomputed here for n <= 10000).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np                                                    # noqa: E402
from scipy.cluster.vq import kmeans2                                  # noqa: E402
from scipy.special import log_ndtr, ndtri                             # noqa: E402
from scipy.stats import qmc                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_a as H                                                  # noqa: E402
import nodes_b as NB                                                  # noqa: E402

LOG2PI = np.log(2.0 * np.pi)


# ------------------------------------------------------------- composites
class Composite:
    """Tabulated distribution of Z = min_j (mu_j + s_j eps_j)."""
    def __init__(self, mu, s, points=4001):
        lo = float((mu - 9 * s).min()); hi = float((mu + 9 * s).max())
        self.z = np.linspace(lo, hi, points)
        zz = (self.z[:, None] - mu[None, :]) / s[None, :]
        logS_j = log_ndtr(-zz)                                          # (points, members)
        self.logS = logS_j.sum(1)
        logphi_j = -0.5 * zz ** 2 - 0.5 * LOG2PI - np.log(s)[None, :]
        # g(z) = S(z) sum_j phi_j(z) / S_j(z)
        self.logf = self.logS + np.log(np.exp(logphi_j - logS_j).sum(1) + 1e-300)
        self.lo, self.hi = lo, hi
        # location summary for the lattice window: where the mass is
        cdf = 1.0 - np.exp(self.logS)
        self.q_lo = float(self.z[np.searchsorted(cdf, 1e-12)]); self.q_hi = float(self.z[min(np.searchsorted(cdf, 1 - 1e-12), points - 1)])

    def eval(self, x):
        """(logS, logf) at x; beyond the table: S=1 left, S=0 right."""
        logS = np.interp(x, self.z, self.logS, left=0.0, right=-np.inf)
        logf = np.interp(x, self.z, self.logf, left=-np.inf, right=-np.inf)
        return logS, logf


def mini_lattice(mu_k, sd_k, V_k, comps, V_c, F, W, points=513):
    """p for kept runners (normal) and composites (tabulated), min wins.
    Same kernel as winning.factor.core.win_probabilities_factor."""
    K = len(mu_k); G = len(comps); n_all = K + G
    p = np.zeros(n_all)
    M_k = mu_k[None, :] + F @ V_k.T                                    # (nodes, K)
    C = F @ V_c.T if G else np.zeros((len(F), 0))                      # composite shifts (nodes, G)
    for a in range(len(F)):
        loc_lo = np.concatenate([M_k[a] - 9 * sd_k, [C[a, g] + comps[g].q_lo for g in range(G)]])
        loc_hi = np.concatenate([M_k[a] + 9 * sd_k, [C[a, g] + comps[g].q_hi for g in range(G)]])
        lo = loc_lo.min(); hi = loc_hi.min() + 0.0                     # beyond the smallest upper edge the field survival is ~0
        hi = max(hi, lo + 1e-6)
        x = np.linspace(lo, hi, points); dx = x[1] - x[0]
        z = (x[None, :] - M_k[a][:, None]) / sd_k[:, None]             # (K, L)
        logS = np.empty((n_all, points)); logf = np.empty((n_all, points))
        logS[:K] = log_ndtr(-z); logf[:K] = -0.5 * z ** 2 - 0.5 * LOG2PI - np.log(sd_k)[:, None]
        for g in range(G):
            logS[K + g], logf[K + g] = comps[g].eval(x - C[a, g])
        logSfield = logS.sum(0)
        rest = np.exp(np.clip(logSfield[None, :] - logS, -745.0, 0.0))
        p += W[a] * dx * (np.exp(np.clip(logf, -745.0, 700.0)) * rest).sum(1)
    return p


def group_members(V_rest, G, how, seed=0):
    if G <= 0 or len(V_rest) == 0:
        return []
    if G >= len(V_rest):
        return [np.array([i]) for i in range(len(V_rest))]
    if how == "kmeans":
        _, lab = kmeans2(V_rest, G, minit="++", seed=seed)
    else:                                                               # seriation: order along the top principal direction, cut contiguously
        Vc = V_rest - V_rest.mean(0); _, _, Wt = np.linalg.svd(Vc, full_matrices=False)
        order = np.argsort(Vc @ Wt[0]); lab = np.empty(len(V_rest), int); lab[order] = np.arange(len(V_rest)) * G // len(V_rest)
    return [np.flatnonzero(lab == g) for g in range(G) if np.any(lab == g)]


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--m", type=int, nargs="+", default=[11, 13])
    ap.add_argument("--groups", type=int, nargs="+", default=[0, 20, 100])
    ap.add_argument("--how", choices=["kmeans", "seriation"], nargs="+", default=["kmeans", "seriation"])
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--points", type=int, default=513)
    ap.add_argument("--validate", action="store_true", help="check the mini lattice against the engine on the kept field (no composites)")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    mu, V, D, sharp = H.make_field(args.n, 3, 0.05, 3); sd = np.sqrt(D)
    truth, d_scr, tkey = NB.largen_truth(args.n)
    order = np.argsort(-truth); targets = list(order[:8]) + [int(np.argmin(np.abs(truth - q))) for q in (1e-2, 3e-3, 1e-3, 3e-4)]
    print(f"\n=== composite n={args.n} rank 3 D 0.05 (sharpness {sharp:.0f}); truth {tkey} certified to {d_scr:.1e} ===", flush=True)
    t = time.perf_counter(); F0 = NB.sobol_nodes(3, 7, 0); keep = NB.prune_field(mu, V, D, F0, args.margin); t_pilot = time.perf_counter() - t
    K = int(keep.sum()); rest = np.flatnonzero(~keep); kept = np.flatnonzero(keep)
    print(f"    pilot 2^7 ({t_pilot:.1f}s): keep {K} runners at margin {args.margin}; truth mass outside {truth[rest].sum():.2e}, largest outside {truth[rest].max() if len(rest) else 0:.1e}; targets all kept: {all(keep[t] for t in targets)}")
    if args.validate:
        F = NB.sobol_nodes(3, 9, 0); W = np.full(len(F), 1 / len(F))
        pe = NB.price(mu[kept], V[kept], D[kept], F); pm = mini_lattice(mu[kept], sd[kept], V[kept], [], np.zeros((0, 3)), F, W, args.points)
        print(f"    validate: mini lattice vs engine on the kept field at 2^9 nodes: max |diff| {np.abs(pm / pm.sum() - pe).max():.1e}, raw sum {pm.sum():.6f}")
    rows = []
    def emit(**kw):
        rows.append(kw)
        if args.csv:
            new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
            with open(args.csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(kw))
                if new: w.writeheader()
                w.writerow(kw)
    print(f"  {'method':<32} {'runners/pass':>12} {'nodes':>6} {'kept max abs':>12} {'targets max rel':>15} {'groups max abs':>14} {'s':>7}", flush=True)
    for how in args.how:
        for G in args.groups:
            if G == 0 and how != args.how[0]:
                continue
            t = time.perf_counter()
            members = group_members(V[rest], G, how)
            comps = []; V_c = np.zeros((len(members), 3)); gidx = []
            for g, mem in enumerate(members):
                idx = rest[mem]; V_g = V[idx].mean(0); V_c[g] = V_g
                s = np.sqrt(D[idx] + ((V[idx] - V_g) ** 2).sum(1))
                comps.append(Composite(mu[idx], s)); gidx.append(idx)
            t_build = time.perf_counter() - t
            for m in args.m:
                F = NB.sobol_nodes(3, m, 0); W = np.full(len(F), 1 / len(F))
                t = time.perf_counter(); p = mini_lattice(mu[kept], sd[kept], V[kept], comps, V_c, F, W, args.points); dt = time.perf_counter() - t
                p = p / p.sum()
                pk = p[:K]; err_k = float(np.abs(pk - truth[kept]).max())
                tmap = {int(i): j for j, i in enumerate(kept)}
                rel = float(max(abs(pk[tmap[t_]] - truth[t_]) / truth[t_] for t_ in targets if t_ in tmap))
                err_g = float(max([abs(p[K + g] - truth[gidx[g]].sum()) for g in range(len(comps))], default=0.0))
                name = f"{'prune' if G == 0 else how + ' G=' + str(len(comps))} Sobol 2^{m}"
                print(f"  {name:<32} {K + len(comps):>12} {2**m:>6} {err_k:>12.1e} {rel:>15.1e} {err_g:>14.1e} {dt:>7.1f}  (+build {t_build:.1f}s)", flush=True)
                emit(n=args.n, method=name, how=how, groups=len(comps), kept=K, runners_per_pass=K + len(comps), nodes=2 ** m, kept_max_abs=err_k, targets_max_rel=rel, groups_max_abs=err_g, mass_outside=float(truth[rest].sum()), seconds=dt, build_seconds=t_build)
    if args.n <= 10000:
        for m in args.m:
            t = time.perf_counter(); pf = NB.price(mu, V, D, NB.sobol_nodes(3, m, 0)); dt = time.perf_counter() - t
            print(f"  {'full field Sobol 2^' + str(m):<32} {args.n:>12} {2**m:>6} {np.abs(pf[kept] - truth[kept]).max():>12.1e} {max(abs(pf[t_] - truth[t_]) / truth[t_] for t_ in targets):>15.1e} {'':>14} {dt:>7.1f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
