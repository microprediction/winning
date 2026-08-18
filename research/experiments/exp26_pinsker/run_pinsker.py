"""Experiment 26: a rigorous certificate for covariance-fit share error.

Winner events depend only on contrasts, so for any basis B of the
mean-zero subspace, with C = B' Sigma B and Chat = B' Sighat B:

    |p_i(Sigma) - p_i(Sighat)| <= TV(N(B'mu, C), N(B'mu, Chat))
                               <= sqrt(KL(N(B'mu,C) || N(B'mu,Chat)) / 2)

by Pinsker, with the Gaussian KL explicit. Measured here in two regimes:
a slowly decaying factor spectrum (bound holds, nearly vacuous) and a
fast-decaying one (bound informative at good fits). The bound is
consistently ~100x conservative but is the first rigorous bridge from
covariance-fit error to share error in this setting.

Run:  python experiments/exp26_pinsker/run_pinsker.py
Output: results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import (  # noqa: E402
    factor_model_projected,
    hermite_nodes,
    win_probabilities_factor,
)

HERE = Path(__file__).resolve().parent
SEED = 4


def kl_contrast(S1, S2, B):
    n = S1.shape[0]
    C1 = B.T @ S1 @ B
    C2 = B.T @ S2 @ B
    C2i = np.linalg.inv(C2)
    _, ld1 = np.linalg.slogdet(C1)
    _, ld2 = np.linalg.slogdet(C2)
    return 0.5 * (np.trace(C2i @ C1) - (n - 1) + ld2 - ld1)


def run(label, scales, rng, rows):
    n = 30
    Vt = rng.normal(0, 1, (n, 4)) * scales
    Dt = rng.uniform(0.4, 1.2, n)
    Sigma = Vt @ Vt.T + np.diag(Dt)
    mu = rng.normal(0, 1.0, n)
    B = np.linalg.qr(np.eye(n) - np.ones((n, n)) / n)[0][:, :n - 1]
    F, W = hermite_nodes(4)
    p_true = win_probabilities_factor(mu, Vt, Dt, F, W)
    for k_fit in (1, 2, 3):
        Vf, Df = factor_model_projected(Sigma, k_fit)
        Sighat = Vf @ Vf.T + np.diag(Df)
        Ff, Wf = hermite_nodes(k_fit)
        p_hat = win_probabilities_factor(mu, Vf, Df, Ff, Wf)
        actual = float(np.abs(p_hat - p_true).max())
        bound = float(np.sqrt(max(kl_contrast(Sigma, Sighat, B), 0) / 2))
        print(f"{label} k_fit={k_fit}: max|dp| {actual:.2e} "
              f"bound {bound:.2e} ratio {actual/bound:.3f}")
        rows.append(f"{label},{k_fit},{actual:.3e},{bound:.3e}")


def main():
    rows = ["regime,k_fit,max_share_err,pinsker_bound"]
    rng = np.random.default_rng(SEED)
    run("flat", np.array([0.5, 0.5, 0.5, 0.5]), rng, rows)
    rng = np.random.default_rng(SEED)
    run("decaying", np.array([0.7, 0.45, 0.08, 0.03]), rng, rows)
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
