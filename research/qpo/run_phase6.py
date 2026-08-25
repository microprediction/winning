"""Phase VI: runtime scaling, and the matched-accuracy frontier.

Two questions, kept apart.

SCALING. On synthetic factor posteriors (mu, V, D) generated directly -- which
is what a sparse GP would hand you, with no N x N covariance ever formed -- how
does each method's wall time grow with N and with r? Factorisation time and
probability time are reported separately, because hiding the first inside
"preprocessing" is how a fast method gets invented. Dense MC-qPO is included
until the N x N covariance stops fitting in memory, which is the whole reason
the qPO implementation prefilters to 10,000 candidates.

FRONTIER. The decisive comparison for the deterministic method is not against
dense Monte Carlo, which it obviously beats, but against well-implemented
Monte Carlo drawn from the same factor model. A factorisation makes sampling
cheap: Y = mu + V z + sqrt(D) eps costs O(N r) per draw. So the question is
whether the deterministic calculation buys anything the sampler cannot buy by
simply drawing more. Both methods are run at several budgets, each measured
against its own self-consistency (two independent seeds/scrambles), and the
answer is read off as time to reach a given accuracy.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import tv_error  # noqa: E402
from pom import (pom_factor_mc, pom_fast, pom_flite, pom_full_mc,  # noqa: E402
                 sobol_nodes)


def synthetic(N: int, r: int, seed: int = 0, spread: float = 1.0):
    """A factor posterior with molecular-looking geometry.

    Marginal variances vary by about 1.6x and the means are spread over about
    one standard deviation, matching what the qm9 snapshot actually looks like,
    because both the lattice width and the flatness of p depend on that.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.40, 1.00, N)
    V = rng.standard_normal((N, r)) / np.sqrt(r) * 0.6
    tot = d + np.sum(V ** 2, axis=1)
    mu = rng.standard_normal(N) * spread * np.sqrt(tot.mean())
    return mu, V, d


def timed(fn, *a, **kw):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    secs = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, secs, peak


def scaling(Ns, ranks, sobol_m, points, mc_samples, dense_limit_gb=4.0):
    rows = []
    for N in Ns:
        for r in ranks:
            mu, V, d = synthetic(N, r)
            F, W = sobol_nodes(r, m=sobol_m, seed=0)
            _, t_fast, m_fast = timed(pom_fast, mu, V, d, F, W, points=points)
            _, t_fmc, m_fmc = timed(pom_factor_mc, mu, V, d, M=mc_samples,
                                    seed=0, chunk=max(1, int(2e7 // N)))
            var = d + np.sum(V ** 2, axis=1)
            _, t_lite, m_lite = timed(pom_flite, mu, var)

            row = {"N": N, "rank": r, "sobol_nodes": len(F), "points": points,
                   "mc_samples": mc_samples,
                   "fast_pom_seconds": t_fast, "fast_peak_mb": m_fast,
                   "fast_candidates_per_second": N / t_fast,
                   "factormc_pom_seconds": t_fmc, "factormc_peak_mb": m_fmc,
                   "flite_seconds": t_lite, "flite_peak_mb": m_lite}

            gb = N * N * 8 / 1e9
            if gb <= dense_limit_gb:
                Sigma, t_build, m_build = timed(
                    lambda: V @ V.T + np.diag(d))
                _, t_dense, m_dense = timed(pom_full_mc, mu, Sigma,
                                            M=10000, seed=0,
                                            chunk=max(1, int(4e7 // N)))
                row.update({"dense_build_seconds": t_build,
                            "dense_mc10k_seconds": t_dense,
                            "dense_peak_mb": max(m_build, m_dense),
                            "dense_cov_gb": gb})
                del Sigma
                gc.collect()
            else:
                row.update({"dense_build_seconds": np.nan,
                            "dense_mc10k_seconds": np.nan,
                            "dense_peak_mb": np.nan, "dense_cov_gb": gb})
            rows.append(row)
            print(f"  N={N:7d} r={r:4d}: fast {t_fast:8.2f}s "
                  f"({N / t_fast:9.0f} cand/s, {m_fast:6.0f}MB) | "
                  f"factorMC({mc_samples:,}) {t_fmc:7.2f}s | "
                  f"F-LITE {t_lite:6.3f}s | "
                  f"dense10k {row['dense_mc10k_seconds'] if not np.isnan(row['dense_mc10k_seconds']) else float('nan'):8.2f}s "
                  f"(cov {gb:.1f}GB)", flush=True)
    return rows


def frontier(N, ranks, points, sobol_ms, mc_list):
    """Accuracy against wall time, each method judged by its own noise."""
    rows = []
    for r in ranks:
        mu, V, d = synthetic(N, r)
        # a converged answer to score both against
        Fh, Wh = sobol_nodes(r, m=max(sobol_ms) + 2, seed=7)
        p_star = pom_fast(mu, V, d, Fh, Wh, points=points)

        for m in sobol_ms:
            F1, W1 = sobol_nodes(r, m=m, seed=0)
            F2, W2 = sobol_nodes(r, m=m, seed=101)
            p1, t1, mem = timed(pom_fast, mu, V, d, F1, W1, points=points)
            p2 = pom_fast(mu, V, d, F2, W2, points=points)
            rows.append({"N": N, "rank": r, "method": "qPO-fast",
                         "budget": len(F1), "seconds": t1, "peak_mb": mem,
                         "self_tv": tv_error(p1, p2),
                         "tv_vs_converged": tv_error(p1, p_star)})
            print(f"  r={r:3d} fast  Q={len(F1):6d} {t1:7.2f}s "
                  f"self_tv={tv_error(p1, p2):.2e} "
                  f"tv*={tv_error(p1, p_star):.2e}", flush=True)

        for M in mc_list:
            p1, t1, mem = timed(pom_factor_mc, mu, V, d, M=M, seed=0,
                                chunk=max(1, int(2e7 // N)))
            p2 = pom_factor_mc(mu, V, d, M=M, seed=101,
                               chunk=max(1, int(2e7 // N)))
            rows.append({"N": N, "rank": r, "method": "qPO-factorMC",
                         "budget": M, "seconds": t1, "peak_mb": mem,
                         "self_tv": tv_error(p1, p2),
                         "tv_vs_converged": tv_error(p1, p_star)})
            print(f"  r={r:3d} fmc   M={M:9d} {t1:7.2f}s "
                  f"self_tv={tv_error(p1, p2):.2e} "
                  f"tv*={tv_error(p1, p_star):.2e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scaling", "frontier", "both"],
                    default="both")
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[1000, 5000, 10000, 50000, 100000, 250000])
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    ap.add_argument("--sobol-m", type=int, default=10)
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--mc-samples", type=int, default=1_000_000)
    ap.add_argument("--frontier-N", type=int, default=10000)
    ap.add_argument("--frontier-ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--frontier-ms", type=int, nargs="+", default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--frontier-mc", type=int, nargs="+",
                    default=[100_000, 300_000, 1_000_000, 3_000_000, 10_000_000,
                             30_000_000])
    args = ap.parse_args()

    if args.mode in ("scaling", "both"):
        print("scaling:")
        rows = scaling(args.Ns, args.ranks, args.sobol_m, args.points,
                       args.mc_samples)
        pd.DataFrame(rows).to_csv(HERE / "results" / "phase6_scaling.csv",
                                  index=False)
    if args.mode in ("frontier", "both"):
        print(f"\nfrontier at N={args.frontier_N}:")
        rows = frontier(args.frontier_N, args.frontier_ranks, args.points,
                        args.frontier_ms, args.frontier_mc)
        pd.DataFrame(rows).to_csv(HERE / "results" / "phase6_frontier.csv",
                                  index=False)
    print("done")


if __name__ == "__main__":
    main()
