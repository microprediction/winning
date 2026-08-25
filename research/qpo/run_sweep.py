"""Phases II, IV, V, VII: does a low-rank factor model preserve the qPO batch?

One posterior, one candidate universe, every method reading the same (mu, Sigma).
For each rank r the oracle factor model Sigma_r = V_r V_r' + D_r is built from
the truncated eigendecomposition, and the resulting acquisition scores are
compared with a high-sample dense Monte Carlo reference.

Three things this script is careful about, because without them the numbers
mislead:

  * The reference is noisy too. Everything is also computed for a second,
    independent reference seed, so every agreement number comes with the value
    that two runs of the reference method itself achieve. That is the ceiling.
    A rank that matches the reference as well as the reference matches itself
    has nothing left to prove.

  * These probabilities are nearly uniform. At N = 500 the largest is about
    three times the smallest, so a batch chosen at random already scores well
    on qPO efficiency. Raw eta is therefore reported next to a normalised
    version that puts random at 0 and optimal at 1.

  * The paper's own default budget is one of the methods. qPO ships with
    10,000 Monte Carlo samples; with N candidates and probabilities near 1/N
    that is about 10,000/N winner counts per candidate. Whether that resolves a
    top-100 is a measurable question, and it is measured here rather than
    assumed.
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

from factorize import (contrast_factor, cov_error, eig_factor,  # noqa: E402
                       effective_rank, project, quotient_cov_error, top_eigen)
from metrics import (batch_agreement, compare, diversity_stats,  # noqa: E402
                     oracle_batch_value, qpo_efficiency, select_batch,
                     spearman, top_set, tv_error)
from pom import (pom_alite, pom_factor_mc, pom_fast, pom_flite,  # noqa: E402
                 pom_full_mc, pom_independent, score_greedy, score_ucb,
                 sobol_nodes)

RANKS = [0, 2, 4, 8, 16, 32, 64, 128, 256]


def nodes_for_rank(r: int, m: int, seed: int = 0):
    if r == 0:
        return None, None
    return sobol_nodes(r, m=m, seed=seed)


def timed(fn, *a, **kw):
    tracemalloc.start()
    t0 = time.time()
    out = fn(*a, **kw)
    secs = time.time() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, secs, peak


def eta_scale(p_ref, b):
    """Random-batch and optimal qPO objective, to put eta on a usable scale."""
    p_ref = np.asarray(p_ref, dtype=float)
    q_opt = float(np.sort(p_ref)[-b:].sum())    # the best any batch can score
    q_rand = float(b * p_ref.mean())
    return q_rand, q_opt


def run_one(snapshot: str, N: int, ranks, batch: int, ref_samples: int,
            fmc_samples: int, sobol_m: int, points: int, seeds=(0, 1),
            qpo_default_seeds=(0, 1, 2, 3, 4), qpo_default_M: int = 10000):
    d0 = HERE / "snapshots" / snapshot / f"N{N}"
    mu = np.load(d0 / "mu.npy")
    Sigma = np.load(d0 / "Sigma.npy")
    fps = np.load(d0 / "fps.npy").astype(float)
    oracle = np.load(d0 / "oracle.npy")
    var = np.diag(Sigma).copy()
    outdir = HERE / "results" / f"{snapshot}_N{N}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {snapshot} N={N} batch={batch} ===")
    if N <= 2000:
        # each of these is a full N^3 eigendecomposition; at N = 10,000 the two
        # of them would cost more than the rest of the sweep, and the top_eigen
        # call below already provides what the rank ladder needs
        print("spectrum (raw)     ", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in effective_rank(Sigma).items()})
        print("spectrum (quotient)", {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in effective_rank(Sigma, True).items()})

    # ---- reference, twice ------------------------------------------------
    refs = {}
    for s in seeds:
        (p, se), secs, peak = timed(pom_full_mc, mu, Sigma, M=ref_samples,
                                    seed=1000 + s,
                                    chunk=max(1, int(4e7 // N)), return_se=True)
        refs[s] = p
        print(f"reference seed {s}: M={ref_samples:,} in {secs:.1f}s "
              f"(TV noise {0.5 * se.sum():.2e}, peak {peak:.0f}MB)")
        np.save(outdir / f"p_full_seed{s}.npy", p)
    p_ref = refs[seeds[0]]
    ref_secs = secs
    q_rand, q_opt = eta_scale(p_ref, batch)
    print(f"eta scale: random batch {q_rand / q_opt:.4f}, optimal 1.0  "
          f"(Q*={q_opt:.4f}, max p={p_ref.max():.2e}, uniform={1 / N:.2e})")

    np.save(outdir / "mu.npy", mu)
    if N <= 2000:
        np.save(outdir / "Sigma.npy", Sigma)

    rows = []

    def record(method, p_test, secs, peak, rank=np.nan, extra=None,
               fac_secs=0.0, n_mc=np.nan, is_probability=True):
        m = compare(p_ref, p_test, b=batch, mu=mu)
        if not is_probability:
            # UCB and Greedy are scores, not probabilities: a distance between
            # them and p has no meaning. Their ranking comparisons still do.
            for k in ("tv_error", "l1_error", "max_abs_error"):
                m[k] = np.nan
        m.update(batch_agreement(p_ref, p_test, (10, 25, 50, 100), mu=mu))
        sel = select_batch(p_test, batch, mu)
        m.update(diversity_stats(fps, sel))
        m.update(oracle_batch_value(oracle, p_test, batch, mu=mu))
        q_test = float(p_ref[sel].sum())
        row = {
            "snapshot": snapshot, "N": N, "method": method, "rank": rank,
            "n_mc_samples": n_mc, "batch": batch,
            "factorization_seconds": fac_secs, "pom_seconds": secs,
            "total_seconds": fac_secs + secs, "peak_memory_mb": peak,
            "qpo_objective": q_test,
            "qpo_efficiency_normalised": (q_test - q_rand) / (q_opt - q_rand),
            **m,
        }
        if extra:
            row.update(extra)
        rows.append(row)
        tv = row["tv_error"]
        print(f"  {method:24s} r={rank!s:>4s} "
              f"eta={row['qpo_efficiency']:.4f} "
              f"etaN={row['qpo_efficiency_normalised']:.3f} "
              f"top100={row['top100_recall']:.2f} "
              f"TV={'   n/a' if not np.isfinite(tv) else format(tv, '.4f')} "
              f"rho={row['spearman']:.3f} "
              f"tan={row['mean_tanimoto']:.3f} {row['total_seconds']:.1f}s")
        return row

    # ---- the reference against itself: the ceiling -----------------------
    if len(seeds) > 1:
        record("reference-2nd-seed", refs[seeds[1]], ref_secs, np.nan,
               n_mc=ref_samples)

    # ---- the shipped qPO default -----------------------------------------
    for s in qpo_default_seeds:
        p, secs, peak = timed(pom_full_mc, mu, Sigma, M=qpo_default_M, seed=s,
                              chunk=max(1, int(4e7 // N)))
        record(f"qPO-MC-{qpo_default_M}", p, secs, peak, n_mc=qpo_default_M,
               extra={"mc_seed": s})

    # ---- ranks ------------------------------------------------------------
    r_max = max(ranks)
    print(f"  eigendecomposing once for r<={r_max} ...", flush=True)
    eig_raw, eig_secs, _ = timed(top_eigen, Sigma, r_max)
    eig_quot, eigq_secs, _ = timed(top_eigen, project(Sigma), r_max)
    print(f"  eigh {eig_secs:.1f}s raw, {eigq_secs:.1f}s quotient")

    for r in ranks:
        for fitname, fitfn, eig in (("eig", eig_factor, eig_raw),
                                    ("contrast", contrast_factor, eig_quot)):
            if r == 0 and fitname == "contrast":
                continue
            (V, dd), fac_secs, fac_peak = timed(fitfn, Sigma, r, eig=eig)
            # the eigendecomposition is the real factorisation cost; charge it
            fac_secs += eig_secs if fitname == "eig" else eigq_secs
            ce = cov_error(Sigma, V, dd) if (fitname == "eig" and N <= 2000) else {}
            qe = quotient_cov_error(Sigma, V, dd) if N <= 2000 else {}
            F, W = nodes_for_rank(r, sobol_m)

            if r == 0:
                p, secs, peak = timed(pom_independent, mu, dd, points=points)
                selferr = 0.0
            else:
                p, secs, peak = timed(pom_fast, mu, V, dd, F, W, points=points)
                # independent scramble at the same budget: the quadrature's own
                # error, so the runtime number is never quoted unconverged
                F2, W2 = sobol_nodes(r, m=sobol_m, seed=101)
                selferr = tv_error(p, pom_fast(mu, V, dd, F2, W2, points=points))
            record(f"qPO-fast-{fitname}", p, secs, peak, rank=r,
                   fac_secs=fac_secs, n_mc=0,
                   extra={**ce, **qe, "sobol_nodes": 0 if r == 0 else len(F),
                          "quad_self_tv": selferr, "min_d": float(dd.min()),
                          "eig_seconds": eig_secs if fitname == "eig" else eigq_secs})

            if fitname == "eig":
                p, secs, peak = timed(pom_factor_mc, mu, V, dd, M=fmc_samples,
                                      seed=7, chunk=max(1, int(2e7 // N)))
                record("qPO-factorMC", p, secs, peak, rank=r,
                       fac_secs=fac_secs, n_mc=fmc_samples, extra={**ce, **qe})

    # ---- independence-based fast baselines --------------------------------
    p, secs, peak = timed(pom_flite, mu, var)
    record("F-LITE", p, secs, peak, rank=0)
    p, secs, peak = timed(pom_alite, mu, var)
    record("A-LITE", p, secs, peak, rank=0)

    # ---- non-PoM acquisition ---------------------------------------------
    p, secs, peak = timed(score_ucb, mu, var)
    record("UCB", p, secs, peak, is_probability=False)
    p, secs, peak = timed(score_greedy, mu)
    record("Greedy", p, secs, peak, is_probability=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep.csv", index=False)
    print(f"wrote {outdir / 'sweep.csv'}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="qm9_gap_seed7")
    ap.add_argument("--N", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--ref-samples", type=int, default=20_000_000)
    ap.add_argument("--fmc-samples", type=int, default=1_000_000)
    ap.add_argument("--sobol-m", type=int, default=10)
    ap.add_argument("--points", type=int, default=257)
    args = ap.parse_args()

    out = []
    for N in args.N:
        out.append(run_one(args.snapshot, N, args.ranks, args.batch,
                           args.ref_samples, args.fmc_samples, args.sobol_m,
                           args.points))
    df = pd.concat(out, ignore_index=True)
    dest = HERE / "results" / f"sweep_{args.snapshot}.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
