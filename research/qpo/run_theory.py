"""Does the sensitivity calculation actually explain the rank ladder?

Plackett's relation says that for any symmetric perturbation of the covariance

    d/dt p(Sigma + t Delta) = (1/2) <Delta, H>,   H = Hessian of p in the means,

and test_theory.py confirms it to a few percent on exactly-representable
perturbations. The question here is whether that first-order term explains what
the rank ladder measured on a real posterior, where the perturbation
Delta_r = Sigma - Sigma_r is not small.

Two failure modes have to be excluded before any conclusion, and a first
version of this script fell into the first of them.

  * TRUNCATION. Delta_r is a full-rank matrix with a nearly flat spectrum, so
    keeping its top few dozen eigendirections keeps only a fraction of it. All
    directions are used here, and the retained Frobenius fraction is reported.

  * NON-LINEARITY. Delta_r is not a small perturbation. So the prediction is
    also tested along a scaled path t*Delta_r for t well below one, where first
    order must work if the identity is right, and the value of t at which it
    stops working is reported rather than assumed.

Run at a modest N: the cost is 2 forward passes per eigendirection per rank.
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

from factorize import eig_factor  # noqa: E402
from pom import pom_fast, pom_full_mc, sobol_nodes  # noqa: E402
from theory import second_directional  # noqa: E402


def predict(mu, V, d, F, W, Delta, points, h=None, n_dirs=None, tol=1e-13):
    """(1/2) <Delta, H> by summing second directional derivatives.

    Returns the prediction and the fraction of Delta's Frobenius mass used.
    """
    lam, U = np.linalg.eigh(0.5 * (Delta + Delta.T))
    total = float(np.sum(lam ** 2))
    order = np.argsort(-np.abs(lam))
    if n_dirs is not None:
        order = order[:n_dirs]
    lam, U = lam[order], U[:, order]
    keep = np.abs(lam) > tol * max(np.abs(lam).max(), 1e-300)
    lam, U = lam[keep], U[:, keep]
    pred = np.zeros(len(mu))
    p0 = pom_fast(mu, V, d, F, W, points=points)
    for m in range(len(lam)):
        u = U[:, m]
        hh = h if h is not None else 0.05 * float(np.sqrt(np.median(d))) / max(
            np.max(np.abs(u)), 1e-300)
        pp = pom_fast(mu + hh * u, V, d, F, W, points=points)
        pm = pom_fast(mu - hh * u, V, d, F, W, points=points)
        pred += 0.5 * lam[m] * (pp - 2.0 * p0 + pm) / hh ** 2
    return pred, float(np.sum(lam ** 2) / max(total, 1e-300)), p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--from-size", type=int, default=500)
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--sobol-m", type=int, default=9)
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--ref-samples", type=int, default=20_000_000)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()

    d0 = HERE / "snapshots" / args.snapshot / f"N{args.from_size}"
    mu = np.load(d0 / "mu.npy")[:args.N].copy()
    Sigma = np.load(d0 / "Sigma.npy")[:args.N, :args.N].copy()
    n = len(mu)
    off = ~np.eye(n, dtype=bool)
    fro_sigma = float(np.sqrt(np.sum(Sigma[off] ** 2)))

    print(f"{args.snapshot}, top {n} candidates by UCB")
    t0 = time.time()
    p_ref = pom_full_mc(Sigma=Sigma, mu=mu, M=args.ref_samples, seed=999,
                        chunk=200_000)
    p_ref2 = pom_full_mc(Sigma=Sigma, mu=mu, M=args.ref_samples, seed=998,
                         chunk=200_000)
    print(f"reference {args.ref_samples:,} samples x2 in {time.time() - t0:.0f}s; "
          f"seed-to-seed TV {0.5 * np.abs(p_ref - p_ref2).sum():.5f}")

    rows = []
    for r in args.ranks:
        V, dd = eig_factor(Sigma, r)
        Delta = Sigma - (V @ V.T + np.diag(dd))
        F, W = sobol_nodes(r, m=args.sobol_m, seed=0)
        t0 = time.time()
        pred, captured, p_r = predict(mu, V, dd, F, W, Delta, args.points)
        secs = time.time() - t0
        actual = p_ref - p_r
        rel = float(np.linalg.norm(pred - actual) / np.linalg.norm(actual))
        corr = float(np.corrcoef(pred, actual)[0, 1])
        row = {"rank": r, "scale": 1.0,
               "frobenius_ratio": float(np.sqrt(np.sum(Delta[off] ** 2))) / fro_sigma,
               "captured": captured,
               "predicted_tv": 0.5 * float(np.abs(pred).sum()),
               "measured_tv": 0.5 * float(np.abs(actual).sum()),
               "prediction_rel_err": rel, "prediction_corr": corr,
               "seconds": secs}
        rows.append(row)
        print(f"  r={r:3d} full Delta: frobenius {row['frobenius_ratio']:.3f} "
              f"captured {captured:.3f}  predicted TV {row['predicted_tv']:.4f} "
              f"measured TV {row['measured_tv']:.4f} corr {corr:+.3f} "
              f"({secs:.0f}s)", flush=True)

    # ---- is it first order at all? scale the perturbation down -------------
    r = args.ranks[0]
    V, dd = eig_factor(Sigma, r)
    Delta = Sigma - (V @ V.T + np.diag(dd))
    F, W = sobol_nodes(r, m=args.sobol_m, seed=0)
    pred1, captured, p_r = predict(mu, V, dd, F, W, Delta, args.points)
    print(f"\nfirst-order check at rank {r}: p(Sigma_r + t Delta) against t*pred")
    for t in args.scales:
        St = (V @ V.T + np.diag(dd)) + t * Delta
        Vt, dt = eig_factor(St, n - 1)          # exact re-representation
        # a full-rank factor model is degenerate for the lattice, so evaluate
        # the perturbed covariance the same way the reference is evaluated
        p_t = pom_full_mc(Sigma=St, mu=mu, M=4_000_000, seed=7, chunk=200_000)
        actual = p_t - p_r
        rel = float(np.linalg.norm(actual - t * pred1) / np.linalg.norm(actual))
        rows.append({"rank": r, "scale": t, "prediction_rel_err": rel,
                     "measured_tv": 0.5 * float(np.abs(actual).sum()),
                     "predicted_tv": 0.5 * float(np.abs(t * pred1).sum()),
                     "prediction_corr": float(np.corrcoef(t * pred1, actual)[0, 1])})
        print(f"  t={t:5.2f}: measured TV {rows[-1]['measured_tv']:.4f}  "
              f"predicted TV {rows[-1]['predicted_tv']:.4f}  "
              f"rel err {rel:.3f}  corr {rows[-1]['prediction_corr']:+.3f}",
              flush=True)

    out = HERE / "results" / f"theory_{args.snapshot}_N{n}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
