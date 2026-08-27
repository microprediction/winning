"""The accuracy/runtime frontier on the real posterior.

Every estimator here is given a dial and turned up: dense Monte Carlo qPO gets
more samples, factor Monte Carlo gets more samples, the deterministic factor
probit gets more quadrature nodes. Each setting is scored against a
high-sample reference on the metric that decides the batch.

The two approximate-model methods have a floor -- rank r cannot represent more
of Sigma than rank r represents -- so their curves flatten. Dense Monte Carlo
has no floor but pays N^2 per sample. Where the flat part of the cheap curve
sits relative to the expensive curve is the entire practical question.

Reading the output: qpo_efficiency is the fraction of the achievable qPO batch
objective kept, and top100_recall is agreement with the reference batch. The
reference's own second seed is printed first and is the ceiling for both.
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

from factorize import contrast_factor, eig_factor, top_eigen, project  # noqa: E402
from metrics import (batch_agreement, qpo_efficiency, select_batch,  # noqa: E402
                     spearman, tv_error)
from pom import (pom_alite, pom_factor_mc, pom_fast, pom_flite,  # noqa: E402
                 pom_full_mc, pom_independent, sobol_nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--ref-samples", type=int, default=40_000_000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--dense-M", type=int, nargs="+",
                    default=[10_000, 30_000, 100_000, 300_000, 1_000_000,
                             3_000_000, 10_000_000])
    ap.add_argument("--factor-M", type=int, nargs="+",
                    default=[10_000, 30_000, 100_000, 300_000, 1_000_000,
                             3_000_000, 10_000_000])
    ap.add_argument("--sobol-ms", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    ap.add_argument("--mc-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    d0 = HERE / "snapshots" / args.snapshot / f"N{args.N}"
    mu = np.load(d0 / "mu.npy")
    Sigma = np.load(d0 / "Sigma.npy")
    var = np.diag(Sigma).copy()
    b = args.batch
    chunk = max(1, int(4e7 // args.N))

    print(f"{args.snapshot} N={args.N}: reference with {args.ref_samples:,} samples")
    t0 = time.perf_counter()
    p_ref = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=999, chunk=chunk)
    ref_secs = time.perf_counter() - t0
    p_ref2 = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=998, chunk=chunk)
    print(f"  {ref_secs:.0f}s each; reference vs itself: "
          f"eta={qpo_efficiency(p_ref, p_ref2, b, mu):.4f} "
          f"top{b}={batch_agreement(p_ref, p_ref2, (b,), mu=mu)[f'top{b}_recall']:.3f} "
          f"TV={tv_error(p_ref, p_ref2):.4f}")

    q_opt = float(np.sort(p_ref)[-b:].sum())
    q_rand = float(b * p_ref.mean())
    rows = []

    def add(method, rank, budget, p, secs, seed=np.nan, extra=None):
        sel = select_batch(p, b, mu)
        q = float(p_ref[sel].sum())
        row = {"snapshot": args.snapshot, "N": args.N, "method": method,
               "rank": rank, "budget": budget, "seed": seed, "seconds": secs,
               "qpo_efficiency": q / q_opt,
               "qpo_efficiency_normalised": (q - q_rand) / (q_opt - q_rand),
               "tv_error": tv_error(p_ref, p),
               "spearman": spearman(p_ref, p),
               **batch_agreement(p_ref, p, (10, 100), mu=mu)}
        if extra:
            row.update(extra)
        rows.append(row)
        return row

    r0 = add("reference-2nd-seed", np.nan, args.ref_samples, p_ref2, ref_secs)

    print("\ndense MC-qPO (the shipped estimator):")
    for M in args.dense_M:
        for s in args.mc_seeds:
            t0 = time.perf_counter()
            p = pom_full_mc(mu, Sigma, M=M, seed=s, chunk=chunk)
            secs = time.perf_counter() - t0
            r = add("qPO-dense-MC", np.nan, M, p, secs, seed=s)
        print(f"  M={M:10,d} {r['seconds']:8.2f}s  eta={r['qpo_efficiency']:.4f} "
              f"top100={r['top100_recall']:.2f} TV={r['tv_error']:.4f}", flush=True)

    eig_raw = top_eigen(Sigma, max(args.ranks))
    eig_quo = top_eigen(project(Sigma), max(args.ranks))
    for rank in args.ranks:
        V, dd = eig_factor(Sigma, rank, eig=eig_raw)
        Vc, dc = contrast_factor(Sigma, rank, eig=eig_quo)
        print(f"\nrank {rank}:")
        for m in args.sobol_ms:
            F, W = sobol_nodes(rank, m=m, seed=0)
            t0 = time.perf_counter()
            p = pom_fast(mu, V, dd, F, W, points=args.points)
            secs = time.perf_counter() - t0
            F2, W2 = sobol_nodes(rank, m=m, seed=101)
            self_tv = tv_error(p, pom_fast(mu, V, dd, F2, W2, points=args.points))
            r = add("qPO-fast", rank, len(F), p, secs,
                    extra={"quad_self_tv": self_tv})
            t0 = time.perf_counter()
            pc = pom_fast(mu, Vc, dc, F, W, points=args.points)
            secs_c = time.perf_counter() - t0
            add("qPO-fast-contrast", rank, len(F), pc, secs_c)
            print(f"  fast Q={len(F):6d} {secs:7.2f}s  eta={r['qpo_efficiency']:.4f} "
                  f"top100={r['top100_recall']:.2f} TV={r['tv_error']:.4f} "
                  f"selfTV={self_tv:.1e}", flush=True)
        for M in args.factor_M:
            t0 = time.perf_counter()
            p = pom_factor_mc(mu, V, dd, M=M, seed=0,
                              chunk=max(1, int(2e7 // args.N)))
            secs = time.perf_counter() - t0
            r = add("qPO-factorMC", rank, M, p, secs, seed=0)
            print(f"  fmc  M={M:9,d} {secs:7.2f}s  eta={r['qpo_efficiency']:.4f} "
                  f"top100={r['top100_recall']:.2f} TV={r['tv_error']:.4f}",
                  flush=True)

    print("\nindependence baselines:")
    for name, fn in (("qPO-independent-exact", lambda: pom_independent(mu, var, points=args.points)),
                     ("F-LITE", lambda: pom_flite(mu, var)),
                     ("A-LITE", lambda: pom_alite(mu, var))):
        t0 = time.perf_counter()
        p = fn()
        secs = time.perf_counter() - t0
        r = add(name, 0, np.nan, p, secs)
        print(f"  {name:24s} {secs:8.4f}s  eta={r['qpo_efficiency']:.4f} "
              f"top100={r['top100_recall']:.2f} TV={r['tv_error']:.4f}")

    out = HERE / "results" / f"budget_{args.snapshot}_N{args.N}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
