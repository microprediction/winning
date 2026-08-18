"""Experiment 14c: eigenbasis replication for the covariance boundary
(fifth/sixth review rounds).

The single-basis Part A confounds spectral decay with basis realization and
with the post-standardization leading eigenvalue. Here: 3 independent
orthogonal eigenbases x 3 decay rates, k in {1, 8}, truth 8e6-draw MC, GHK
at R=1e4 at the exact Sigma. Reports the range across bases.

Run:  python experiments/exp14_boundaries/run_basis_replication.py
Output: results_bases.csv
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import factor_model_contrast, hermite_nodes, qmc_nodes  # noqa: E402
from run_boundaries import factor_shares_base, spectral_corr  # noqa: E402
from run_ghk_benchmark import ghk_prob  # noqa: E402

HERE = Path(__file__).resolve().parent
N = 50


def main():
    rng = np.random.default_rng(77)
    mu = rng.normal(0, 1.0, N)
    rows = ["basis,gamma,quantity,value"]
    summary = {}
    for b in range(3):
        basis, _ = np.linalg.qr(rng.standard_normal((N, N)))
        for gamma in (0.5, 1.5, 3.0):
            C, eig = spectral_corr(N, gamma, basis)
            L = np.linalg.cholesky(C + 1e-10 * np.eye(N))
            counts = np.zeros(N)
            r2 = np.random.default_rng(9 + b)
            for _ in range(40):
                X = mu[:, None] + L @ r2.standard_normal((N, 200_000))
                counts += np.bincount(np.argmin(X, axis=0), minlength=N)
            truth = counts / counts.sum()
            errs = {}
            for k in (1, 8):
                V, D = factor_model_contrast(C, k)
                Fk, Wk = hermite_nodes(k) if k <= 4 else qmc_nodes(k)
                pk, _ = factor_shares_base(mu, V, D, Fk, Wk)
                errs[k] = np.abs(pk - truth).max()
            pg = np.array([ghk_prob(-mu, C, i, R=10_000, seed=100 + i)
                           for i in range(N)])
            pg = pg / pg.sum()
            eg = np.abs(pg - truth).max()
            print(f"basis {b} gamma={gamma}: lead eig {eig[0]:.1f}, "
                  f"k=1 {errs[1]:.1e}, k=8 {errs[8]:.1e}, GHK R=1e4 {eg:.1e}")
            rows += [f"{b},{gamma},k1,{errs[1]:.3e}",
                     f"{b},{gamma},k8,{errs[8]:.3e}",
                     f"{b},{gamma},ghk1e4,{eg:.3e}",
                     f"{b},{gamma},lead_eig,{eig[0]:.2f}"]
            summary.setdefault(gamma, []).append((errs[8], eg))
    print("\nacross bases (k=8 vs GHK R=1e4):")
    for gamma, vals in summary.items():
        k8 = [v[0] for v in vals]; gh = [v[1] for v in vals]
        wins = sum(1 for a, g in vals if a < g)
        print(f"  gamma={gamma}: k8 range [{min(k8):.1e}, {max(k8):.1e}], "
              f"GHK range [{min(gh):.1e}, {max(gh):.1e}], k8 wins {wins}/3")
    (HERE / "results_bases.csv").write_text("\n".join(rows) + "\n")
    print("wrote results_bases.csv")


if __name__ == "__main__":
    main()
