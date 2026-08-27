"""Where does first order stop working? Measured with common random numbers.

A first attempt at this compared p(Sigma_r + t Delta) against p(Sigma_r) using
independent Monte Carlo runs. At small t the true change is smaller than the
Monte Carlo noise, so the check reported nonsense exactly where first order
should look best. Both ends of the difference are now taken from the SAME
random stream: the same standard normals pushed through two nearby Choleskys.
The sampling error then largely cancels and the difference is resolvable far
below the level either estimate could be resolved on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factorize import eig_factor  # noqa: E402
from pom import _chol, sobol_nodes  # noqa: E402
from run_theory import predict  # noqa: E402


def counts_crn(mu, Sigma, M, seed, chunk=200_000):
    """Winner counts using a fixed standard-normal stream."""
    A = _chol(Sigma)
    rng = np.random.default_rng(seed)
    n = len(mu)
    counts = np.zeros(n)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        Z = rng.standard_normal((m, n))
        np.add.at(counts, np.argmax(mu[None, :] + Z @ A.T, axis=1), 1)
        done += m
    return counts / M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--from-size", type=int, default=500)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--sobol-m", type=int, default=9)
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--M", type=int, default=40_000_000)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.05, 0.1, 0.25, 0.5, 1.0])
    args = ap.parse_args()

    d0 = HERE / "snapshots" / args.snapshot / f"N{args.from_size}"
    mu = np.load(d0 / "mu.npy")[:args.N].copy()
    Sigma = np.load(d0 / "Sigma.npy")[:args.N, :args.N].copy()
    V, dd = eig_factor(Sigma, args.rank)
    S_r = V @ V.T + np.diag(dd)
    Delta = Sigma - S_r
    F, W = sobol_nodes(args.rank, m=args.sobol_m, seed=0)

    print(f"rank {args.rank}, N={args.N}, common random numbers, M={args.M:,}")
    pred, captured, _ = predict(mu, V, dd, F, W, Delta, args.points)
    print(f"  first-order prediction built from {captured:.3f} of Delta")

    base = counts_crn(mu, S_r, args.M, seed=4242)
    # noise floor of the CRN difference: two independent streams at t=0
    base2 = counts_crn(mu, S_r, args.M, seed=4243)
    floor = 0.5 * float(np.abs(base - base2).sum())
    print(f"  CRN difference noise floor (t=0): TV {floor:.5f}")

    rows = []
    for t in args.scales:
        pt = counts_crn(mu, S_r + t * Delta, args.M, seed=4242)
        actual = pt - base
        rel = float(np.linalg.norm(actual - t * pred) / np.linalg.norm(actual))
        corr = float(np.corrcoef(t * pred, actual)[0, 1])
        rows.append({"rank": args.rank, "scale": t,
                     "measured_tv": 0.5 * float(np.abs(actual).sum()),
                     "predicted_tv": 0.5 * float(np.abs(t * pred).sum()),
                     "rel_err": rel, "corr": corr, "noise_floor_tv": floor})
        print(f"  t={t:5.2f}: measured TV {rows[-1]['measured_tv']:.5f}  "
              f"predicted TV {rows[-1]['predicted_tv']:.5f}  "
              f"rel err {rel:.3f}  corr {corr:+.3f}", flush=True)

    out = HERE / "results" / f"theory_scaling_{args.snapshot}_N{args.N}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
