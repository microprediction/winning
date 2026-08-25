"""Phase III. Separate the two errors, and settle the quadrature budget.

Two things are checked here, in this order, before any accuracy claim about
qPO is made.

1. Is the fast probability calculation right? Fast probit and factor Monte
   Carlo are run on the SAME factor model (mu, V_r, D_r), so the covariance
   approximation error is identically zero and any disagreement is a bug. The
   test statistic is the z-score (p_fast - p_fmc)/se, not total variation:
   these probabilities are all near 1/N, so total variation is swamped by
   Monte Carlo noise while the z-score is not.

2. How many factor nodes does rank r need? The r-dimensional quadrature is the
   only free knob in the fast method, and its cost is linear in the node count,
   so the runtime claim is meaningless without a converged node count. Reported
   as self-convergence: total variation between Q and 2Q nodes.

Writes results/phase3_<snapshot>.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factorize import eig_factor  # noqa: E402
from metrics import tv_error  # noqa: E402
from pom import pom_factor_mc, pom_fast, sobol_nodes  # noqa: E402


def q_convergence(mu, V, d, r, m_list, points, seed=0):
    """Total variation between successive Sobol node counts."""
    out = []
    prev = None
    for m in m_list:
        F, W = sobol_nodes(r, m=m, seed=seed)
        t0 = time.time()
        p = pom_fast(mu, V, d, F, W, points=points)
        secs = time.time() - t0
        row = {"nodes": len(F), "seconds": secs,
               "tv_vs_previous": np.nan if prev is None else tv_error(prev, p)}
        # independent scramble at the same budget: a second error estimate
        F2, W2 = sobol_nodes(r, m=m, seed=seed + 101)
        p2 = pom_fast(mu, V, d, F2, W2, points=points)
        row["tv_scramble_pair"] = tv_error(p, p2)
        out.append(row)
        prev = p
    return out, prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 8, 32, 64, 128])
    ap.add_argument("--mc", type=int, default=4_000_000)
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--m-list", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11])
    args = ap.parse_args()

    d0 = HERE / "snapshots" / args.snapshot / f"N{args.N}"
    mu = np.load(d0 / "mu.npy")
    Sigma = np.load(d0 / "Sigma.npy")
    print(f"{args.snapshot} N={args.N}  mu spread {np.ptp(mu):.4f}  "
          f"sd {np.sqrt(np.diag(Sigma)).min():.4f}-{np.sqrt(np.diag(Sigma)).max():.4f}")

    rows = []
    for r in args.ranks:
        V, d = eig_factor(Sigma, r)
        conv, p_fast = q_convergence(mu, V, d, r, args.m_list, args.points)
        print(f"\nrank {r}: min(d)={d.min():.3e}")
        for c in conv:
            print(f"   Q={c['nodes']:6d}  {c['seconds']:7.2f}s  "
                  f"tv_vs_prev={c['tv_vs_previous']:.3e}  "
                  f"tv_scramble_pair={c['tv_scramble_pair']:.3e}")

        t0 = time.time()
        p_fmc, se = pom_factor_mc(mu, V, d, M=args.mc, seed=11,
                                  chunk=max(1, int(2e7 // len(mu))), return_se=True)
        fmc_secs = time.time() - t0
        z = (p_fast - p_fmc) / np.maximum(se, 1e-300)
        finite = np.isfinite(z)
        print(f"   factor MC M={args.mc:,} in {fmc_secs:.1f}s  "
              f"max|z|={np.max(np.abs(z[finite])):.2f}  "
              f"mean z={np.mean(z[finite]):+.3f}  sd z={np.std(z[finite]):.3f}  "
              f"TV={tv_error(p_fast, p_fmc):.4e}  "
              f"(MC TV noise floor {0.5 * se.sum():.4e})")

        for c in conv:
            rows.append({"snapshot": args.snapshot, "N": args.N, "rank": r,
                         "points": args.points, **c})
        rows.append({"snapshot": args.snapshot, "N": args.N, "rank": r,
                     "points": args.points, "nodes": np.nan,
                     "seconds": fmc_secs, "mc_samples": args.mc,
                     "max_abs_z": float(np.max(np.abs(z[finite]))),
                     "mean_z": float(np.mean(z[finite])),
                     "sd_z": float(np.std(z[finite])),
                     "tv_fast_vs_fmc": tv_error(p_fast, p_fmc),
                     "tv_mc_noise_floor": float(0.5 * se.sum())})

    out = HERE / "results" / f"phase3_{args.snapshot}_N{args.N}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
