"""Experiment 24: the per-alternative factor-conditioned RQMC baseline
(tenth review round) -- the strongest fair adversary.

Conditioning on the factors f AND the alternative's own shock z gives, for
each alternative separately,

  p_i = E_{f,z} prod_{j != i} Phi((mu_i - mu_j + (v_i - v_j)'f
                                   + sqrt(D_i) z) / sqrt(D_j)),

a (k+1)-dimensional integral regardless of N, evaluable by scrambled-Sobol
RQMC. This uses exactly the factor structure the paper's method assumes, so
it is the obvious modern competitor. Its cost for the full share vector is
O(R N^2); the shared survival field computes the same N integrals coupled,
in O(QN(k+L)). This experiment measures that gap directly.

Identity verified against the lattice map (1.7e-5 at 2^14 points, N=8)
before any comparison.

Run:  python experiments/exp24_factor_rqmc/run_factor_rqmc.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402
from run_ghk_benchmark import make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def factor_rqmc_shares(mu, V, D, m, seed=11):
    """All N shares by per-alternative (k+1)-dim scrambled-Sobol RQMC."""
    n, k = V.shape
    X = qmc.MultivariateNormalQMC(np.zeros(k + 1), np.eye(k + 1),
                                  seed=seed).random(2**m)
    f, z = X[:, :k], X[:, k]
    sD = np.sqrt(D)
    p = np.zeros(n)
    for i in range(n):
        others = np.arange(n) != i
        args = (mu[i] - mu[others][None, :]
                + f @ (V[i] - V[others]).T
                + sD[i] * z[:, None]) / sD[others][None, :]
        p[i] = np.mean(np.prod(ndtr(args), axis=1))
    return p / p.sum()


def main():
    rng = np.random.default_rng(SEED)
    rows = ["N,method,setting,seconds,max_err"]
    F, W = hermite_nodes(2)
    for n in (50, 200, 1000):
        mu, V, D = make_problem(n, 2, rng, spread=1.0 if n <= 200 else 1.5)
        truth = mc_shares(mu, V, D, 2_000_000, seed=9)
        t0 = time.perf_counter()
        p_lat = win_probabilities_factor(-mu, V, D, F, W)
        t_lat = time.perf_counter() - t0
        e_lat = np.abs(p_lat - truth).max()
        print(f"N={n}: lattice {t_lat*1000:6.0f} ms, err {e_lat:.1e}")
        rows.append(f"{n},lattice,default,{t_lat:.3f},{e_lat:.3e}")
        for m in (12, 14):
            t0 = time.perf_counter()
            p_q = factor_rqmc_shares(mu, V, D, m)  # max-wins directly
            t_q = time.perf_counter() - t0
            e_q = np.abs(p_q - truth).max()
            print(f"        factor-RQMC 2^{m}: {t_q:6.1f} s, err {e_q:.1e} "
                  f"({t_q/t_lat:.0f}x lattice time)")
            rows.append(f"{n},factor_rqmc,2^{m},{t_q:.3f},{e_q:.3e}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
