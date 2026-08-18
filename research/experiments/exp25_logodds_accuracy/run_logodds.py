"""Experiment 25: the lattice's true log-odds accuracy, share by share.

Production settings (L=501, Gauss-Hermite 15^2) against a super-converged
reference (L=5001, Gauss-Hermite 60^2) on the same model, so the error
measured is the lattice's own, not a simulation reference's noise. Shares
are bucketed by magnitude to show the error is uniform in log-odds down to
deep tails, which no simulation reference could certify: matching relative
error eps on a share p costs 1/(p eps^2) draws.

Run:  python experiments/exp25_logodds_accuracy/run_logodds.py
Output: results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 3
BUCKETS = ((1e-2, 1.0), (1e-4, 1e-2), (1e-6, 1e-4), (1e-9, 1e-6))


def dense_hermite(order=60):
    from numpy.polynomial.hermite_e import hermegauss
    x1, w1 = hermegauss(order)
    w1 = w1 / w1.sum()
    F = np.array([[a, b] for a in x1 for b in x1])
    W = np.array([wa * wb for wa in w1 for wb in w1])
    return F, W


def main():
    Fd, Wd = dense_hermite()
    # at N=1000 use GH 40^2 pruned + L=3001 (cheaper, still far beyond
    # production resolution; pruning at 1e-12 of max weight)
    Fd2, Wd2 = dense_hermite(40)
    keep = Wd2 > 1e-12 * Wd2.max()
    Fd2, Wd2 = Fd2[keep], Wd2[keep] / Wd2[keep].sum()
    rows = ["N,seed,bucket_lo,bucket_hi,count,max_log_odds_err"]
    for n, seeds in ((20, (3,)), (100, (3,)), (1000, (3, 4, 5))):
      for seed in seeds:
        rng = np.random.default_rng(seed)
        mu = rng.normal(0, 1.5, n)
        V = rng.normal(0, 0.5 / np.sqrt(2), (n, 2))
        D = rng.uniform(0.5, 1.5, n)
        p_prod = win_probabilities_factor(mu, V, D, *hermite_nodes(2),
                                          points=501)
        Fr, Wr = (Fd, Wd) if n <= 100 else (Fd2, Wd2)
        Lr = 5001 if n <= 100 else 3001
        p_true = win_probabilities_factor(mu, V, D, Fr, Wr, points=Lr)
        rel = np.abs(np.log(p_prod) - np.log(p_true))
        for lo, hi in BUCKETS:
            m = (p_true >= lo) & (p_true < hi)
            if m.any():
                print(f"N={n} seed {seed}: shares in [{lo:.0e},{hi:.0e}): "
                      f"{m.sum():>4} max log-odds err {rel[m].max():.2e}")
                rows.append(f"{n},{seed},{lo:.0e},{hi:.0e},{int(m.sum())},"
                            f"{rel[m].max():.3e}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
