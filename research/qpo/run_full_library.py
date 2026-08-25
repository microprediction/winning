"""qPO over the entire library, against qPO over the 10,000 the prefilter keeps.

This is the comparison the released implementation cannot make. acquire()
routes any pool larger than 10,000 candidates through a UCB prefilter first,
because sampling a dense N-dimensional Gaussian costs O(N^2) per draw and the
covariance alone would be 143 GB at the full QM9 library. The factor route
never forms that covariance: the GP hands over (mu, V, D) directly, and the
probability of maximality is a quadrature over r factors.

So the interesting question is not "10k prefiltered Monte Carlo versus 10k
fast" but

    10k-prefiltered MC-qPO   versus   full-library fast qPO,

and specifically whether the full-library batch contains molecules the
prefilter threw away, and whether those molecules are any good. The held-out
oracle values answer the second half.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factorgp import factor_posterior, load_gp  # noqa: E402
from metrics import diversity_stats, select_batch  # noqa: E402
from pom import pom_fast, pom_flite, pom_full_mc, sobol_nodes  # noqa: E402


def timed(fn, *a, **kw):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    secs = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, secs, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--inducing", type=int, default=512)
    ap.add_argument("--sobol-m", type=int, default=8)
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--prefilter", type=int, default=10000)
    ap.add_argument("--mc-samples", type=int, default=10000)
    ap.add_argument("--mc-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-elements", type=float, default=2.4e7)
    args = ap.parse_args()

    snap = HERE / "snapshots" / args.snapshot
    post, meta = load_gp(snap)
    c = meta.get("c", 1)
    b = args.batch

    fps_all = np.load(meta["fps_cache"], mmap_mode="r")
    idx = np.load(snap / "full_fp_index.npy")
    oracle = np.load(snap / "full_oracle.npy")
    mu_full_saved = np.load(snap / "full_mu.npy")
    var_full = np.load(snap / "full_var.npy")
    N = len(idx)
    print(f"{args.snapshot}: full candidate pool N = {N:,}, "
          f"dense covariance would be {N * N * 8 / 1e9:.0f} GB")

    X = np.ascontiguousarray(fps_all[idx])          # (N, 2048) float32
    print(f"  fingerprints {X.shape} {X.dtype} ({X.nbytes / 1e9:.2f} GB)")

    # the prefiltered universe the released code would actually score
    ucb = mu_full_saved + np.sqrt(var_full)
    order = np.argsort(-ucb)
    keep = order[:args.prefilter]
    true_top1 = np.argsort(-oracle)[:max(1, N // 100)]
    true_top001 = np.argsort(-oracle)[:max(1, N // 10000)]

    rows = []

    def record(method, sel, secs, peak, extra=None):
        sel = np.asarray(sel)
        row = {"snapshot": args.snapshot, "N_scored": extra.pop("N_scored", N),
               "method": method, "seconds": secs, "peak_memory_mb": peak,
               "batch": b,
               "batch_oracle_mean": float(oracle[sel].mean()),
               "batch_oracle_max": float(oracle[sel].max()),
               "batch_oracle_top10": float(np.sort(oracle[sel])[-10:].mean()),
               "n_in_true_top1pct": int(np.isin(sel, true_top1).sum()),
               "n_in_true_top0.01pct": int(np.isin(sel, true_top001).sum()),
               "frac_outside_prefilter": float(np.mean(~np.isin(sel, keep))),
               **diversity_stats(X, sel)}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {method:34s} {secs:8.1f}s  oracle(mean {row['batch_oracle_mean']:.4f}"
              f", top10 {row['batch_oracle_top10']:.4f}, max {row['batch_oracle_max']:.4f})"
              f"  top1%hits={row['n_in_true_top1pct']:3d}"
              f"  outside prefilter={row['frac_outside_prefilter']:.2f}"
              f"  tanimoto={row['mean_tanimoto']:.3f}", flush=True)
        return row

    # ---------------- the released pipeline: prefilter, then dense MC -------
    print("\n10k-prefiltered dense MC-qPO (the released pipeline):")
    Xk = np.ascontiguousarray(X[keep])
    (Sig_k), t_cov, m_cov = timed(post.covariance, Xk.astype(np.float64))
    mu_k = c * post.mean(Xk)
    if c == -1:
        pass  # covariance is unchanged by the sign flip
    print(f"  forming the 10k covariance: {t_cov:.1f}s, {m_cov:.0f}MB, "
          f"{Sig_k.nbytes / 1e9:.2f} GB")
    for s in args.mc_seeds:
        p, secs, peak = timed(pom_full_mc, mu_k, Sig_k, M=args.mc_samples,
                              seed=s, chunk=4000)
        sel_local = select_batch(p, b, mu_k)
        record(f"prefiltered-MC-qPO-{args.mc_samples}", keep[sel_local],
               secs + t_cov, max(peak, m_cov),
               extra={"N_scored": args.prefilter, "mc_seed": s,
                      "cov_seconds": t_cov})
    del Sig_k

    # a like-for-like fast run on the same 10k, to separate prefilter from method
    for rank in args.ranks:
        (mu_p, V_p, d_p), t_fac, m_fac = timed(
            factor_posterior, post, Xk, rank=rank, inducing=args.inducing, seed=0)
        F, W = sobol_nodes(rank, m=args.sobol_m, seed=0)
        p, secs, peak = timed(pom_fast, c * mu_p, V_p, d_p, F, W,
                              points=args.points, max_elements=args.max_elements)
        sel_local = select_batch(p, b, c * mu_p)
        record(f"prefiltered-fast-qPO-r{rank}", keep[sel_local], secs + t_fac,
               max(peak, m_fac),
               extra={"N_scored": args.prefilter, "rank": rank,
                      "factorization_seconds": t_fac, "sobol_nodes": len(F)})

    # ---------------- the whole library ------------------------------------
    print(f"\nfull-library fast qPO over all {N:,} candidates:")
    for rank in args.ranks:
        (out), t_fac, m_fac = timed(factor_posterior, post, X, rank=rank,
                                    inducing=args.inducing, seed=0,
                                    return_info=True)
        mu_f, V_f, d_f, info = out
        F, W = sobol_nodes(rank, m=args.sobol_m, seed=0)
        p, secs, peak = timed(pom_fast, c * mu_f, V_f, d_f, F, W,
                              points=args.points, max_elements=args.max_elements)
        sel = select_batch(p, b, c * mu_f)
        record(f"full-library-fast-qPO-r{rank}", sel, secs + t_fac,
               max(peak, m_fac),
               extra={"rank": rank, "factorization_seconds": t_fac,
                      "sobol_nodes": len(F), "pom_seconds": secs,
                      "n_floored": info["n_floored"],
                      "max_p": float(p.max())})
        np.save(HERE / "results" / f"p_fulllib_r{rank}_{args.snapshot}.npy", p)

    # ---------------- full-library baselines that also scale ---------------
    p, secs, peak = timed(pom_flite, c * mu_full_saved, var_full)
    record("full-library-F-LITE", select_batch(p, b, c * mu_full_saved),
           secs, peak, extra={"rank": 0})
    sel = select_batch(ucb, b)
    record("full-library-UCB", sel, 0.0, 0.0, extra={"rank": np.nan})
    sel = select_batch(c * mu_full_saved, b)
    record("full-library-Greedy", sel, 0.0, 0.0, extra={"rank": np.nan})

    df = pd.DataFrame(rows)
    out = HERE / "results" / f"full_library_{args.snapshot}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
