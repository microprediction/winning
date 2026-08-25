"""Milestone 1: do one-factor conditional envelopes beat raw winner counting?

Not inversion. The question that has to be settled first is whether, for a
fixed dense Gaussian race, integrating one direction analytically estimates p
and J enough more efficiently than counting winners to be worth doing.

Both estimators are run at the same number of draws on the same covariance,
repeatedly, and the per-component variance ratio is recorded. Two normalisations
are reported and they answer different questions:

  per draw    -- how much information conditioning extracts from one sample
  per second  -- whether that survives the extra work, which is the only
                 version that decides anything

The second is the one an earlier experiment in this directory failed. Rao-
Blackwellising the dense reference by integrating the whole idiosyncratic block
cost O(N L) per draw and lost badly on wall clock despite winning per draw.
Integrating ONE direction costs O(N log N) against the O(N^2) already spent
drawing the residual, so the arithmetic should come out the other way. That is
the claim being tested.

Covariance families deliberately include cases no small factor model can
represent: a slowly decaying spectrum, clusters of near substitutes, and
condition numbers to 1e8.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from covfamilies import ABILITY, FAMILIES, abilities, sqrt_psd  # noqa: E402
from envelope import raw_shares, rb_shares_batch, split_one_factor  # noqa: E402


def replicate_variance(fn, reps, seed0):
    """Component-wise variance of an estimator across independent replicates."""
    est = np.array([fn(seed0 + 1000 * k) for k in range(reps)])
    return est.mean(axis=0), est.var(axis=0, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 200, 500])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--regimes", nargs="+", default=list(ABILITY))
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="envelope_variance.csv")
    args = ap.parse_args()

    rows = []
    for n in args.n:
        for fam in args.families:
            Sigma = FAMILIES[fam](n, args.seed)
            b, R, A, info = split_one_factor(Sigma)
            S_sqrt = sqrt_psd(Sigma)
            cond = float(np.linalg.cond(Sigma + 1e-12 * np.eye(n)))
            for regime in args.regimes:
                mu = abilities(n, ABILITY[regime], args.seed)

                t0 = time.perf_counter()
                m_rb, v_rb = replicate_variance(
                    lambda s: rb_shares_batch(mu, b, A, M=args.M, seed=s)[0],
                    args.reps, 11)
                t_rb = (time.perf_counter() - t0) / args.reps

                t0 = time.perf_counter()
                m_raw, v_raw = replicate_variance(
                    lambda s: raw_shares(mu, S_sqrt, M=args.M, seed=s)[0],
                    args.reps, 11)
                t_raw = (time.perf_counter() - t0) / args.reps

                ok = v_rb > 0
                ratio = np.full(n, np.nan)
                ratio[ok] = v_raw[ok] / v_rb[ok]
                # per-second comparison: variance scales as 1/M, so equal-time
                # variance is variance * time
                per_sec = ratio * (t_raw / t_rb)
                bias = float(0.5 * np.abs(m_rb - m_raw).sum())

                row = {"n": n, "family": fam, "regime": regime, "M": args.M,
                       "reps": args.reps, "cond": cond,
                       "factor_trace_fraction": info["trace_fraction"],
                       "loading_spread": info["loading_spread"],
                       "median_var_ratio": float(np.nanmedian(ratio)),
                       "mean_var_ratio": float(np.nanmean(ratio)),
                       "total_var_ratio": float(v_raw.sum() / max(v_rb.sum(), 1e-300)),
                       "median_var_ratio_per_second": float(np.nanmedian(per_sec)),
                       "total_var_ratio_per_second": float(
                           v_raw.sum() / max(v_rb.sum(), 1e-300) * (t_raw / t_rb)),
                       "seconds_rb": t_rb, "seconds_raw": t_raw,
                       "tv_between_estimators": bias,
                       "max_p": float(m_raw.max())}
                rows.append(row)
                print(f"  n={n:5d} {fam:14s} {regime:9s} "
                       f"var ratio: median {row['median_var_ratio']:8.1f} "
                       f"total {row['total_var_ratio']:8.1f} | "
                       f"per second {row['total_var_ratio_per_second']:8.1f} "
                       f"(rb {t_rb * 1e3:6.1f}ms raw {t_raw * 1e3:6.1f}ms) "
                       f"agree {bias:.4f}", flush=True)

    df = pd.DataFrame(rows)
    dest = HERE / "results" / args.out
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nby family (total variance ratio, per draw and per second):")
    print(df.groupby("family")[["total_var_ratio",
                                "total_var_ratio_per_second"]].median().to_string(
        float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
