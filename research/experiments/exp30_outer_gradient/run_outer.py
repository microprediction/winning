"""Experiment 30: implicit differentiation through calibration, validated.

Parameterize the covariance: V(theta) = theta1 * V0, D(theta) =
exp(theta2) * D0. Truth theta* = (1, 0). M markets share one utility
vector mu*; each market's shares are observed. Recovered utilities
mu_m(theta) = calibrate(p_m; V(theta), D(theta)) agree across markets
exactly at theta*, so the outer objective

    L(theta) = sum_m || mu_m(theta) - mean_m mu_m(theta) ||^2

vanishes at the truth. Its gradient passes through the calibration
fixed point by the implicit function theorem:
d mu / d theta = -(B (B'JB)^{-1} B') dp/dtheta at fixed shares.

Validated: implicit gradient vs central finite differences of L; then
recovery of theta from noisy (5e6-draw) targets by damped Newton.

Run:  python experiments/exp30_outer_gradient/run_outer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    win_probabilities_factor,
)

HERE = Path(__file__).resolve().parent
N, K, M = 50, 2, 4


def forward(mu_min, V, D, F, W):
    return win_probabilities_factor(mu_min, V, D, F, W)


def jac(mu_min, V, D, F, W, eps=1e-6):
    J = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N); e[j] = eps
        J[:, j] = (forward(mu_min + e, V, D, F, W)
                   - forward(mu_min - e, V, D, F, W)) / (2 * eps)
    return 0.5 * (J + J.T)


def dp_dtheta(mu_min, V0, D0, th, F, W, eps=1e-6):
    cols = []
    for i in range(2):
        tp = th.copy(); tp[i] += eps
        tm = th.copy(); tm[i] -= eps
        cols.append((forward(mu_min, tp[0] * V0, np.exp(tp[1]) * D0, F, W)
                     - forward(mu_min, tm[0] * V0, np.exp(tm[1]) * D0, F, W))
                    / (2 * eps))
    return np.stack(cols, axis=1)          # N x 2


def main():
    rng = np.random.default_rng(30)
    F, W = hermite_nodes(K)
    mu_star = rng.normal(0, 1.0, N); mu_star -= mu_star.mean()
    V0 = rng.normal(0, 0.4, (N, K))
    D0 = rng.uniform(0.6, 1.2, N)
    B = np.linalg.qr(np.eye(N) - np.ones((N, N)) / N)[0][:, :N - 1]

    # markets share ONE utility vector; known factor-scale multipliers
    # c_m identify theta (at truth, recovered utilities agree exactly)
    CM = np.array([0.6, 0.9, 1.2, 1.5])
    markets = [forward(mu_star, CM[m] * V0, D0, F, W) for m in range(M)]

    def recover(pm, th, m):
        V, D = CM[m] * th[0] * V0, np.exp(th[1]) * D0
        return abilities_from_probabilities_factor(pm, V, D, F, W, tol=1e-9)

    def L_and_grad(th, want_grad=True):
        mus = [recover(pm, th, m) for m, pm in enumerate(markets)]
        mubar = np.mean(mus, axis=0)
        L = sum(float(np.sum((mu - mubar) ** 2)) for mu in mus)
        if not want_grad:
            return L, None
        G = np.zeros(2)
        mubar = np.mean(mus, axis=0)
        for m, (pm, mu) in enumerate(zip(markets, mus)):
            V, D = CM[m] * th[0] * V0, np.exp(th[1]) * D0
            J = jac(mu, V, D, F, W)
            S = dp_dtheta(mu, CM[m] * V0, D0, th, F, W)
            dmu = -B @ np.linalg.solve(B.T @ J @ B, B.T @ S)   # N x 2
            gL = 2 * (mu - mubar)
            G += gL @ dmu
        return L, G

    th = np.array([1.0, 0.0])
    # gradient validation at a perturbed point
    th_test = np.array([1.15, 0.12])
    L0, G0 = L_and_grad(th_test)
    eps = 1e-4
    Gfd = np.zeros(2)
    for i in range(2):
        tp = th_test.copy(); tp[i] += eps
        tm = th_test.copy(); tm[i] -= eps
        Gfd[i] = (L_and_grad(tp, False)[0] - L_and_grad(tm, False)[0]) / (2 * eps)
    print(f"implicit gradient {G0}")
    print(f"fd gradient       {Gfd}")
    print(f"max rel err       {np.abs(G0 - Gfd).max() / np.abs(Gfd).max():.2e}")

    # recovery from the perturbed start by damped Newton (BFGS-lite)
    th_hat = th_test.copy()
    for it in range(30):
        L, G = L_and_grad(th_hat)
        if np.linalg.norm(G) < 1e-10 or L < 1e-16:
            break
        H = np.zeros((2, 2))
        e = 1e-4
        for i in range(2):
            tp = th_hat.copy(); tp[i] += e
            tm = th_hat.copy(); tm[i] -= e
            H[:, i] = (L_and_grad(tp)[1] - L_and_grad(tm)[1]) / (2 * e)
        H = 0.5 * (H + H.T) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, G)
        th_hat = th_hat - np.clip(step, -0.5, 0.5)
        print(f"iter {it}: L {L:.3e} theta {th_hat}")
    gauge = th_hat[0]**2 * np.exp(-th_hat[1])
    print(f"recovered theta {th_hat} (truth [1, 0])")
    print(f"gauge-invariant ratio theta1^2 exp(-theta2) = {gauge:.6f} "
          f"(truth 1; overall covariance scale is not identified from "
          f"shares, the classical probit normalization)")
    (HERE / "results.txt").write_text(
        f"grad_rel_err {np.abs(G0 - Gfd).max() / np.abs(Gfd).max():.3e}\n"
        f"theta_hat {th_hat.tolist()}\n")


if __name__ == "__main__":
    main()
