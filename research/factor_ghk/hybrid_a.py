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
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")

import numpy as np                                                    # noqa: E402
from scipy.optimize import linprog                                    # noqa: E402
from scipy.special import ndtri                                       # noqa: E402
from scipy.stats import norm, qmc                                     # noqa: E402

import winning.factor as wf                                           # noqa: E402
from winning.factor.core import hermite_nodes, qmc_nodes              # noqa: E402


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


def lattice_conditional(mu, V, D, f, points):
    """Exact conditional race at factor value f: one lattice pass, all runners."""
    return np.asarray(wf.race_probabilities(mu + V @ f, D=D, points=points), dtype=float)


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
def hybrid_a(mu, V, D, R, points, margin, seed=7):
    """Per-winner GHK in factor space.  Returns (p_raw, lattice_passes)."""
    n, k = V.shape
    U = qmc.Sobol(d=k, scramble=True, seed=seed).random(R)           # common draws
    p = np.zeros(n); passes = 0
    for i in range(n):
        A, b = competitive_constraints(mu, V, D, i, margin)
        acc = 0.0
        for row in range(R):
            f = np.zeros(k); w = 1.0; ok = True
            for kk in range(k):
                iv = projection_interval(A, b, f[:kk], kk)
                if iv is None:
                    ok = False; break
                f[kk], mass = truncated_draw(iv[0], iv[1], U[row, kk])
                w *= mass
            if not ok or w <= 0:
                continue
            acc += w * lattice_conditional(mu, V, D, f, points)[i]
            passes += 1
        p[i] = acc / R
    return p, passes


def hybrid_a_shared(mu, V, D, R, points, margin, seed=7):
    """Shared draws from a mixture of the winners' truncated proposals; one
    lattice pass per draw serves every winner.  Returns (p_raw, passes)."""
    n, k = V.shape
    U = qmc.Sobol(d=k + 1, scramble=True, seed=seed).random(R)
    cons = [competitive_constraints(mu, V, D, i, margin) for i in range(n)]
    p = np.zeros(n); passes = 0
    for row in range(R):
        i = min(int(U[row, k] * n), n - 1)                            # mixture component
        A, b = cons[i]
        f = np.zeros(k); q_i = 1.0; ok = True
        for kk in range(k):
            iv = projection_interval(A, b, f[:kk], kk)
            if iv is None:
                ok = False; break
            f[kk], mass = truncated_draw(iv[0], iv[1], U[row, kk])
            q_i *= 1.0 / mass                                          # density ratio phi/q along this chain
        if not ok:
            continue
        # mixture density q(f) = (1/n) sum_j phi(f) * [f in poly_j] / mass_j(f-chain); the
        # chain masses depend on the path, so evaluate q_mix by recomputing each j's chain mass at f
        q_mix = 0.0
        for j in range(n):
            Aj, bj = cons[j]
            if np.all(Aj @ f <= bj + 1e-12):
                wj = 1.0
                for kk in range(k):
                    iv = projection_interval(Aj, bj, f[:kk], kk)
                    if iv is None:
                        wj = 0.0; break
                    wj *= norm.cdf(iv[1]) - norm.cdf(iv[0])
                if wj > 0:
                    q_mix += 1.0 / wj
        q_mix /= n
        if q_mix <= 0:
            continue
        p += lattice_conditional(mu, V, D, f, points) / q_mix
        passes += 1
    return p / R, passes


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--D", type=float, nargs="+", default=[0.05])
    ap.add_argument("--R", type=int, nargs="+", default=[16, 64, 256], help="hybrid draws")
    ap.add_argument("--paths", type=int, default=2_000_000)
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--margin", type=float, default=4.0, help="competitive margin, idiosyncratic sd units")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    rows = []
    for D_level in args.D:
        mu, V, D, sharp = make_field(args.n, args.rank, D_level, args.seed)
        t0 = time.perf_counter(); truth = mc_truth(mu, V, D, args.paths); t_truth = time.perf_counter() - t0
        se = float(np.sqrt(truth.max() * (1 - truth.max()) / args.paths))
        print(f"\n=== rank {args.rank}, n={args.n}, D={D_level}: sharpness {sharp:.1f}; truth {args.paths} paths in {t_truth:.0f}s, se ~{se:.1e} ===")
        print(f"  {'method':<40} {'passes':>7} {'max|p-truth|':>13} {'raw sum':>8} {'s':>6}")

        def report(name, p, passes, dt, raw_sum=None):
            err = float(np.abs(p - truth).max())
            print(f"  {name:<40} {passes:>7} {err:>13.1e} {('' if raw_sum is None else f'{raw_sum:8.4f}'):>8} {dt:>6.1f}")
            rows.append(dict(rank=args.rank, n=args.n, D=D_level, sharp=sharp, method=name, passes=passes, err=err, seconds=dt))

        t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, points=args.points)); report("winning default rule", p, -1, time.perf_counter() - t)
        for Q in (15, 41) if args.rank >= 2 else (15, 51, 201):
            F, W = hermite_nodes(args.rank, Q=Q)
            t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F, W=W, points=args.points)); report(f"fixed Gauss-Hermite Q={Q}", p, len(F), time.perf_counter() - t)
        for m in (7, 9, 11):
            F, W = qmc_nodes(args.rank, m=m)
            t = time.perf_counter(); p = np.asarray(wf.race_probabilities(mu, V=V, D=D, F=F, W=W, points=args.points)); report(f"fixed scrambled Sobol 2^{m}", p, len(F), time.perf_counter() - t)
        for R in args.R:
            t = time.perf_counter(); p, passes = hybrid_a(mu, V, D, R, args.points, args.margin); dt = time.perf_counter() - t
            report(f"Hybrid A per-winner R={R} (raw)", p, passes, dt, p.sum()); report(f"Hybrid A per-winner R={R} (normalised)", p / p.sum(), passes, dt)
            t = time.perf_counter(); p, passes = hybrid_a_shared(mu, V, D, R * args.n, args.points, args.margin); dt = time.perf_counter() - t
            report(f"Hybrid A shared R={R * args.n} (raw)", p, passes, dt, p.sum()); report(f"Hybrid A shared R={R * args.n} (normalised)", p / p.sum(), passes, dt)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
