"""Experiment 17: convergence sweeps, numerical diagnostics, and the
error-versus-wall-time frontier (third referee round).

  A. L sweep (lattice resolution) at fixed large Q: self-convergence against
     the finest lattice, plus the pre-normalization defect |1 - 1^T p_raw|.
     Also the per-node-interval variant (each factor node gets its own
     8-sigma window) versus the global interval.
  B. Q sweep (factor quadrature) at fixed L: Gauss-Hermite order sweep at
     k=2; at k=8, scrambled-Sobol RQMC across 8 INDEPENDENT scrambles at
     several point counts -- the distribution across scrambles is reported,
     not one fixed realization.
  C. Jacobian diagnostics at N=50: dense J assembled from N JVP calls;
     symmetry defect, row-sum (translation-invariance) defect, and the
     eigenvalue range of the reduced matrix -B^T J B (min-wins J is minus a
     weighted graph Laplacian, so the reduced matrix must be positive
     definite).
  D. Error-versus-wall-time frontier at N=200, k=2: lattice at several
     (L, GH order); plain direct utility simulation; scrambled-Sobol QMC
     direct simulation (dimension k+N); GHK at several R. Truth: twin
     5*10^7-draw MC references (their half-difference estimates the noise
     floor). Timings are medians of 3 for sub-second methods.

Run:  python experiments/exp17_convergence/run_convergence.py
Outputs: results.csv, figures/frontier_full.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import (  # noqa: E402
    factor_model,
    hermite_nodes,
    jacobian_vector_product,
    qmc_nodes,
    win_probabilities_factor,
)
from run_ghk_benchmark import ghk_all_shares, make_problem, mc_shares  # noqa: E402

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    THREAD_NOTE = "threadpoolctl: BLAS limited to 1 thread"
except ImportError:
    THREAD_NOTE = "threadpoolctl unavailable; thread count not enforced"

HERE = Path(__file__).resolve().parent
SEED = 21


def timed(fn, reps=3):
    best = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best.append(time.perf_counter() - t0)
    return out, float(np.median(best))


def main():
    print(THREAD_NOTE)
    rows = [f"part,quantity,value", f"meta,thread_note,{THREAD_NOTE}"]
    rng = np.random.default_rng(SEED)

    # ---- A: L sweep -----------------------------------------------------------
    print("\nPart A: lattice-resolution sweep (N=200, k=2, GH order 15)")
    mu, V, D = make_problem(200, 2, rng, spread=1.0)
    F, W = hermite_nodes(2)
    Ls = [101, 201, 375, 751, 1501, 3001, 6001]
    ref, _ = win_probabilities_factor(-mu, V, D, F, W, points=12001,
                                      return_total=True)
    print(f"{'L':>6} {'vs L=12001':>12} {'|1-total|':>11} {'per-node vs global':>19}")
    for L in Ls:
        p, tot = win_probabilities_factor(-mu, V, D, F, W, points=L,
                                          return_total=True)
        pn = win_probabilities_factor(-mu, V, D, F, W, points=L,
                                      per_node_interval=True)
        print(f"{L:>6} {np.abs(p - ref).max():>12.1e} {abs(1 - tot):>11.1e} "
              f"{np.abs(p - pn).max():>19.1e}")
        rows += [f"A,L{L}_selfconv,{np.abs(p-ref).max():.3e}",
                 f"A,L{L}_prenorm_defect,{abs(1-tot):.3e}",
                 f"A,L{L}_pernode_diff,{np.abs(p-pn).max():.3e}"]

    # ---- B: Q sweep -----------------------------------------------------------
    print("\nPart B: factor-quadrature sweep at L=1501")
    print("  k=2, GH order sweep (vs order 21):")
    Fr, Wr = hermite_nodes(2, Q=21)
    pref = win_probabilities_factor(-mu, V, D, Fr, Wr)
    for Qo in (3, 5, 7, 9, 11, 15):
        Fq, Wq = hermite_nodes(2, Q=Qo)
        pq = win_probabilities_factor(-mu, V, D, Fq, Wq)
        print(f"    order {Qo:>2} ({len(Wq):>3} nodes): {np.abs(pq-pref).max():.1e}")
        rows.append(f"B,k2_GH{Qo},{np.abs(pq-pref).max():.3e}")

    print("  k=8 (N=50 dense-Sigma fit), RQMC across 8 independent scrambles:")
    n8 = 50
    mu8 = rng.normal(0, 1, n8)
    A8 = rng.standard_normal((n8, n8))
    C8 = A8 @ A8.T
    d8 = np.sqrt(np.diag(C8)); C8 = C8 / np.outer(d8, d8)
    V8, D8 = factor_model(C8, 8)
    Fbig, Wbig = qmc_nodes(8, m=15, seed=999)
    p8ref = win_probabilities_factor(-mu8, V8, D8, Fbig, Wbig)
    for m in (9, 11, 13):
        errs, errs_pn = [], []
        for sc in range(8):
            Fs, Ws = qmc_nodes(8, m=m, seed=sc)
            ps = win_probabilities_factor(-mu8, V8, D8, Fs, Ws)
            pspn = win_probabilities_factor(-mu8, V8, D8, Fs, Ws,
                                            per_node_interval=True)
            errs.append(np.abs(ps - p8ref).max())
            errs_pn.append(np.abs(pspn - p8ref).max())
        print(f"    2^{m:>2} nodes: median {np.median(errs):.1e} "
              f"range [{min(errs):.1e}, {max(errs):.1e}]  "
              f"(per-node interval median {np.median(errs_pn):.1e})")
        rows += [f"B,k8_m{m}_median,{np.median(errs):.3e}",
                 f"B,k8_m{m}_min,{min(errs):.3e}",
                 f"B,k8_m{m}_max,{max(errs):.3e}",
                 f"B,k8_m{m}_pernode_median,{np.median(errs_pn):.3e}"]

    # ---- C: Jacobian diagnostics ---------------------------------------------
    print("\nPart C: Jacobian diagnostics (N=50, k=2, dense J from JVPs)")
    mu50, V50, D50 = make_problem(50, 2, rng, spread=1.0)
    J = np.column_stack([
        jacobian_vector_product(-mu50, V50, D50, F, W, e)
        for e in np.eye(50)])
    sym = np.abs(J - J.T).max()
    rowsum = max(np.abs(J.sum(0)).max(), np.abs(J.sum(1)).max())
    B50 = np.linalg.qr(np.eye(50) - 1.0 / 50)[0][:, :49]
    ev = np.linalg.eigvalsh(-B50.T @ J @ B50)   # min-wins J = -weighted Laplacian
    print(f"  symmetry defect {sym:.1e}; row-sum defect {rowsum:.1e}; "
          f"reduced -B^T J B eigs in [{ev.min():.2e}, {ev.max():.2e}] (PD: {ev.min() > 0})")
    rows += [f"C,symmetry_defect,{sym:.3e}", f"C,rowsum_defect,{rowsum:.3e}",
             f"C,reduced_min_eig,{ev.min():.4e}", f"C,reduced_max_eig,{ev.max():.4e}"]

    # ---- D: error-vs-wall-time frontier --------------------------------------
    print("\nPart D: error vs wall time, N=200, k=2 (truth: twin 5e7-draw MC)")
    ta = mc_shares(mu, V, D, 50_000_000, seed=301)
    tb = mc_shares(mu, V, D, 50_000_000, seed=302)
    truth = 0.5 * (ta + tb)
    noise = np.abs(ta - tb).max()
    print(f"  reference noise scale (twin half-difference): {noise:.1e}")
    rows.append(f"D,truth_noise,{noise:.3e}")

    frontier = []

    for L, Qo in ((375, 5), (751, 9), (1501, 15), (3001, 15)):
        Fq, Wq = hermite_nodes(2, Q=Qo)
        p, dt = timed(lambda: win_probabilities_factor(-mu, V, D, Fq, Wq, points=L))
        frontier.append(("lattice", f"L={L},GH{Qo}", dt, np.abs(p - truth).max()))

    for R in (100_000, 1_000_000, 10_000_000):
        p, dt = timed(lambda: mc_shares(mu, V, D, R, seed=77), reps=1)
        frontier.append(("direct MC", f"R={R:.0e}", dt, np.abs(p - truth).max()))

    dim = 2 + 200
    for m in (14, 17, 20):
        def qmc_direct(m=m):
            sob = qmc.Sobol(d=dim, scramble=True, seed=55)
            counts = np.zeros(200)
            todo = 2**m
            while todo > 0:
                blk = min(todo, 2**14)
                Z = ndtri(np.clip(sob.random(blk), 1e-15, 1 - 1e-15))
                U = mu[None, :] + Z[:, :2] @ V.T + np.sqrt(D)[None, :] * Z[:, 2:]
                counts += np.bincount(np.argmax(U, 1), minlength=200)
                todo -= blk
            return counts / counts.sum()
        p, dt = timed(qmc_direct, reps=1)
        frontier.append(("QMC direct", f"2^{m}", dt, np.abs(p - truth).max()))

    for R in (1000, 10_000):
        p, dt = timed(lambda: ghk_all_shares(mu, V, D, R=R), reps=1)
        frontier.append(("GHK", f"R={R}", dt, np.abs(p - truth).max()))

    # QMC-GHK: scrambled-Sobol uniforms fed into the same GHK kernel
    from run_ghk_benchmark import ghk_prob
    Sig = V @ V.T + np.diag(D)
    for R in (1024, 8192):
        def qmc_ghk(R=R):
            u = qmc.Sobol(d=199, scramble=True, seed=13).random(R)
            pg = np.array([ghk_prob(mu, Sig, i, u=u) for i in range(200)])
            return pg / pg.sum()
        p, dt = timed(qmc_ghk, reps=1)
        frontier.append(("QMC-GHK", f"R={R}", dt, np.abs(p - truth).max()))

    # GHK seed band at R=1000: error distribution over 8 seeds
    errs_band = []
    for sd_ in range(8):
        pg = ghk_all_shares(mu, V, D, R=1000, seed=1000 + 37 * sd_)
        errs_band.append(np.abs(pg - truth).max())
    print(f"  GHK R=1000 seed band (8 seeds): median {np.median(errs_band):.1e} "
          f"range [{min(errs_band):.1e}, {max(errs_band):.1e}]")
    rows += [f"D,ghk_seedband_median,{np.median(errs_band):.3e}",
             f"D,ghk_seedband_min,{min(errs_band):.3e}",
             f"D,ghk_seedband_max,{max(errs_band):.3e}"]

    print(f"  {'method':>12} {'setting':>12} {'seconds':>9} {'max err':>9}")
    for meth, lab, dt, err in frontier:
        print(f"  {meth:>12} {lab:>12} {dt:>9.2f} {err:>9.1e}")
        rows.append(f"D,{meth.replace(' ', '_')}_{lab.replace(',', '_')},{dt:.3f}s {err:.3e}")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    marks = {"lattice": ("o-", "#c2410c"), "direct MC": ("s-", "#5b7c99"),
             "QMC direct": ("d-", "#2a9d8f"), "GHK": ("^-", "#9a9a9a"),
             "QMC-GHK": ("v-", "#6a5acd")}
    for meth in marks:
        pts = sorted((dt, err) for m_, l_, dt, err in frontier if m_ == meth)
        mk, c = marks[meth]
        ax.loglog([a for a, _ in pts], [b for _, b in pts], mk, color=c,
                  label=meth, ms=5)
    ax.axhline(noise, color="#bbbbbb", ls=":", label="truth noise floor")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("max abs share error")
    ax.set_title("Accuracy-time frontier, N=200, k=2 (all methods timed)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "frontier_full.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
    orders = [3, 5, 7, 9, 11, 15]
    gh_errs = [float(r.split(",")[-1]) for r in rows
               if r.startswith("B,k2_GH")]
    gh_nodes = [len(hermite_nodes(2, Q=o)[1]) for o in orders]
    ax2.loglog(gh_nodes, gh_errs, "o-", color="#c2410c",
               label="k=2 Gauss-Hermite refinement")
    ms = [9, 11, 13]
    med = [float(next(r.split(",")[-1] for r in rows
                      if r.startswith(f"B,k8_m{m}_median"))) for m in ms]
    lo = [float(next(r.split(",")[-1] for r in rows
                     if r.startswith(f"B,k8_m{m}_min"))) for m in ms]
    hi = [float(next(r.split(",")[-1] for r in rows
                     if r.startswith(f"B,k8_m{m}_max"))) for m in ms]
    nodes8 = [2**m for m in ms]
    ax2.loglog(nodes8, med, "s-", color="#5b7c99",
               label="k=8 RQMC (median of 8 scrambles)")
    ax2.fill_between(nodes8, lo, hi, color="#5b7c99", alpha=0.2,
                     label="k=8 RQMC scramble range")
    ref_n = np.array([30., 10000.])
    ax2.loglog(ref_n, 3e-2 * ref_n**-0.5, ":", color="#9a9a9a",
               label=r"$Q^{-1/2}$ guide")
    ax2.set_xlabel("factor nodes Q")
    ax2.set_ylabel("max abs share error vs refined reference")
    ax2.set_title("Factor-quadrature refinement at fixed L", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(HERE / "figures" / "refinement.png", dpi=150)

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("\nwrote results.csv, figures/frontier_full.png, figures/refinement.png")


if __name__ == "__main__":
    main()
