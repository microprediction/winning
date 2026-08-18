"""Experiment 16: benchmark addendum — outputs the paper references, plus the
direct-simulation baseline (second-round referee items).

  A. Error metrics beyond max-coordinate: mean absolute, total variation, and
     max over shares > 1e-3, for the exp13 problem sequence (fresh 2e6 truths).
  B. DIRECT UTILITY MONTE CARLO baseline: the natural full-share-vector
     simulator (argmax over utility draws). At each N, its accuracy at the
     wall-time of the lattice transform — the fair forward-speed comparison
     the per-alternative GHK baseline cannot provide.
  C. Inversion replication: three independent MC target simulations at N=1000,
     k=2; utility recovery error and identified-alternative counts per
     replicate; recovery-error-versus-target-share curve.

Run:  python experiments/exp16_benchmark_addendum/run_addendum.py
Outputs: results.csv, figures/recovery_vs_share.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import abilities_from_probabilities_factor, hermite_nodes  # noqa: E402
from run_ghk_benchmark import lattice_shares, make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21   # exp13's seed: reproduce its problem sequence exactly


def metrics(p, truth):
    d = np.abs(p - truth)
    big = truth > 1e-3
    order = np.argsort(truth)[::-1]
    mass99 = order[:int(np.searchsorted(np.cumsum(truth[order]), 0.99)) + 1]
    return {"max": d.max(), "mean": d.mean(), "tv": 0.5 * d.sum(),
            "max_big": d[big].max() if big.any() else np.nan,
            "max_mass99": d[mass99].max()}


def main():
    rng = np.random.default_rng(SEED)
    rows = ["part,N,quantity,value"]

    # ---- A + B: metrics and the direct-MC baseline ------------------------------
    print("Parts A+B: error metrics and direct utility-simulation baseline")
    print(f"{'N':>6} {'lat max':>9} {'lat mean':>9} {'lat TV':>8} "
          f"{'lat s':>7} {'directMC max @ same time':>25}")
    for n in (5, 20, 50, 200, 1000):
        mu, V, D = make_problem(n, 2, rng, spread=1.0 if n <= 200 else 1.5)
        truth = mc_shares(mu, V, D, 2_000_000, seed=9)
        t0 = time.perf_counter()
        p_lat = lattice_shares(mu, V, D)
        t_lat = time.perf_counter() - t0
        m = metrics(p_lat, truth)
        # direct MC with the SAME wall time as the lattice call
        t0 = time.perf_counter()
        draws_per_sec_probe = 200_000
        t_probe0 = time.perf_counter()
        _ = mc_shares(mu, V, D, draws_per_sec_probe, seed=1)
        rate = draws_per_sec_probe / (time.perf_counter() - t_probe0)
        n_draws = max(50_000, int(rate * t_lat))
        p_mc = mc_shares(mu, V, D, n_draws, seed=2)
        m_mc = metrics(p_mc, truth)
        print(f"{n:>6} {m['max']:>9.1e} {m['mean']:>9.1e} {m['tv']:>8.1e} "
              f"{t_lat:>7.2f} {m_mc['max']:>13.1e} ({n_draws/1e6:.1f}M draws)")
        for k_, v_ in m.items():
            rows.append(f"A,{n},lattice_{k_},{v_:.3e}")
        rows.append(f"A,{n},lattice_seconds,{t_lat:.3f}")
        rows.append(f"B,{n},directmc_max_at_matched_time,{m_mc['max']:.3e}")
        rows.append(f"B,{n},directmc_draws,{n_draws}")

    # ---- C: inversion replication ------------------------------------------------
    print("\nPart C: inversion replication (N=1000, k=2, 3 independent targets)")
    n = 1000
    mu, V, D = make_problem(n, 2, rng, spread=1.2)
    mu -= mu.mean()
    F, W = hermite_nodes(2)
    all_rec, all_shares = [], []
    for rep in range(3):
        target = mc_shares(mu, V, D, 5_000_000, seed=40 + rep)
        target = np.maximum(target, 1e-7); target /= target.sum()
        t0 = time.perf_counter()
        a_hat = abilities_from_probabilities_factor(target, V, D, F, W)
        t_inv = time.perf_counter() - t0
        util_hat = -a_hat
        good = target > 3e-4
        err = np.abs((util_hat - util_hat[good].mean()) - (mu - mu[good].mean()))
        rec = err[good].max()
        print(f"  replicate {rep}: {t_inv:.0f}s, identified {good.sum()}/{n} "
              f"(share mass {target[good].sum():.3f}), recovery max {rec:.4f}, "
              f"median {np.median(err[good]):.4f}")
        rows += [f"C,{n},rep{rep}_seconds,{t_inv:.1f}",
                 f"C,{n},rep{rep}_identified,{int(good.sum())}",
                 f"C,{n},rep{rep}_recovery_max,{rec:.5f}",
                 f"C,{n},rep{rep}_recovery_median,{np.median(err[good]):.5f}"]
        all_rec.append(err); all_shares.append(target)

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for rep, (err, tgt) in enumerate(zip(all_rec, all_shares)):
        ax.loglog(tgt, err, ".", ms=3, alpha=0.4, label=f"replicate {rep}")
    ax.axvline(3e-4, color="#9a9a9a", ls=":", label="identification threshold")
    ax.set_xlabel("target share")
    ax.set_ylabel("utility recovery error")
    ax.set_title("Recovery error vs target share, N=1000 (3 independent targets)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "recovery_vs_share.png", dpi=150)
    print("\nwrote results.csv, figures/recovery_vs_share.png")


if __name__ == "__main__":
    main()
