"""Experiment 22: minimax-tilting baseline (Botev 2017), the stronger
Gaussian-probability method every review since round two has requested.

Botev's estimator computes P(l < X < u) for Gaussian X by exponentially
tilting the sequential (GHK-style) proposal: the tilting parameter solves a
saddle-point system chosen to minimize the worst-case log-likelihood ratio,
after which importance sampling has dramatically lower variance than plain
GHK, especially for tail probabilities.

Here each choice probability P(alternative i wins) is the standard
difference-coordinate orthant probability P(d < 0), d = M_i U, and the
tilted estimator is run per alternative (like GHK, it prices one
alternative at a time).

Validation before comparison (the repository's standing anchor rule):
  - N=2 closed form; N=5 vs 10^7-draw Monte Carlo.
Benchmark: error and wall time at N=50 and N=200 vs the exp17 frontier
truth protocol (twin 5e7-draw references).

Run:  python experiments/exp22_minimax_tilting/run_tilting.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import log_ndtr, ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402
from run_ghk_benchmark import make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def _psi_grad(par, L, u):
    """Saddle system for Botev's minimax tilting of P(X < u), X = L Z.

    Unknowns: x[0:n-1] (saddle point) and mu[0:n-1] (tilting means), with
    x[n-1] = mu[n-1] = 0. With sequential bounds
    ub_t = (u_t - sum_{j<t} L_tj x_j) / L_tt and hazard
    h_t = phi(ub_t - mu_t)/Phi(ub_t - mu_t), the first-order conditions are
        x_j - mu_j + h_j = 0                     (x is the tilted mean)
        mu_j + sum_{t>j} (L_tj/L_tt) h_t = 0.
    """
    n = len(u)
    x = np.zeros(n)
    mu = np.zeros(n)
    x[:n - 1] = par[:n - 1]
    mu[:n - 1] = par[n - 1:]
    ub = (u - L @ x + np.diag(L) * x) / np.diag(L)
    lnP = log_ndtr(ub - mu)
    h = np.exp(-0.5 * (ub - mu) ** 2 - 0.5 * np.log(2 * np.pi) - lnP)
    grad_x = x[:n - 1] - mu[:n - 1] + h[:n - 1]
    Lr = L / np.diag(L)[:, None]
    strict = np.tril(np.ones((n, n)), -1)          # t > j strictly
    cross = (Lr * strict).T @ h                    # sum_{t>j} (L_tj/L_tt) h_t
    grad_mu = mu[:n - 1] + cross[:n - 1]
    return np.concatenate([grad_x, grad_mu])


def _psi(x, mu, L, u):
    ub = (u - L @ x + np.diag(L) * x) / np.diag(L)
    return float(np.sum(log_ndtr(ub - mu) + 0.5 * mu**2 - x * mu))


def tilted_orthant(Sigma_d, R, seed):
    """Botev-style estimate of P(d < 0), d ~ N(0, Sigma_d)."""
    n = len(Sigma_d)
    # variance-reduction reorder (smallest bound first is moot at u=0; keep)
    L = np.linalg.cholesky(Sigma_d + 1e-12 * np.eye(n))
    scale = np.sqrt(np.diag(Sigma_d))
    u = np.zeros(n)                     # upper bounds in original scale
    Ls = L / scale[:, None] * 0 + L     # keep L as is; u = 0 already scaled
    sol = root(_psi_grad, np.zeros(2 * (n - 1)), args=(Ls, u), method="hybr",
               options={"maxfev": 4000})
    x = np.zeros(n); mu = np.zeros(n)
    x[:n - 1] = sol.x[:n - 1]
    mu[:n - 1] = sol.x[n - 1:]
    # importance sampling with tilted sequential proposals
    rng = np.random.default_rng(seed)
    Z = np.zeros((R, n))
    lw = np.zeros(R)
    for t in range(n):
        ubt = (u[t] - Z[:, :t] @ Ls[t, :t]) / Ls[t, t]
        a = ndtr(ubt - mu[t])
        zt = mu[t] + ndtri(np.clip(rng.random(R) * a, 1e-300, 1 - 1e-16))
        Z[:, t] = zt
        lw += log_ndtr(ubt - mu[t]) + 0.5 * mu[t] ** 2 - zt * mu[t]
    m = lw.max()
    return float(np.exp(m) * np.mean(np.exp(lw - m)))


def tilted_shares(mu_util, V, D, R, seed):
    n = len(mu_util)
    Sigma = V @ V.T + np.diag(D)
    p = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        M = np.zeros((n - 1, n))
        M[np.arange(n - 1), others] = 1.0
        M[:, i] -= 1.0
        Sd = M @ Sigma @ M.T
        mean_d = mu_util[others] - mu_util[i]
        # P(d + mean_d < 0) = P(d < -mean_d): shift bounds via u = -mean
        nn = n - 1
        L = np.linalg.cholesky(Sd + 1e-12 * np.eye(nn))
        u = -mean_d
        sol = root(_psi_grad, np.zeros(2 * (nn - 1)), args=(L, u),
                   method="hybr", options={"maxfev": 4000})
        x = np.zeros(nn); mt = np.zeros(nn)
        x[:nn - 1] = sol.x[:nn - 1]
        mt[:nn - 1] = sol.x[nn - 1:]
        rng = np.random.default_rng(seed + i)
        Z = np.zeros((R, nn))
        lw = np.zeros(R)
        for t in range(nn):
            ubt = (u[t] - Z[:, :t] @ L[t, :t]) / L[t, t]
            a = ndtr(ubt - mt[t])
            zt = mt[t] + ndtri(np.clip(rng.random(R) * a, 1e-300, 1 - 1e-16))
            Z[:, t] = zt
            lw += log_ndtr(ubt - mt[t]) + 0.5 * mt[t] ** 2 - zt * mt[t]
        m = lw.max()
        p[i] = np.exp(m) * np.mean(np.exp(lw - m))
    return p / p.sum()


def main():
    rng = np.random.default_rng(SEED)
    rows = ["quantity,value"]

    print("anchors")
    mu2 = np.array([0.3, -0.2]); V2 = np.array([[0.6], [-0.1]])
    D2 = np.array([0.8, 1.2])
    var_diff = (V2[0, 0] - V2[1, 0]) ** 2 + D2[0] + D2[1]
    exact = ndtr((mu2[0] - mu2[1]) / np.sqrt(var_diff))
    p2 = tilted_shares(mu2, V2, D2, 50_000, seed=3)
    print(f"  N=2 closed form: tilting err {abs(p2[0]-exact):.2e}")
    rows.append(f"n2_err,{abs(p2[0]-exact):.3e}")

    mu5, V5, D5 = make_problem(5, 2, rng)
    truth5 = mc_shares(mu5, V5, D5, 10_000_000)
    p5 = tilted_shares(mu5, V5, D5, 50_000, seed=5)
    print(f"  N=5 vs 1e7 MC: tilting err {np.abs(p5-truth5).max():.2e}")
    rows.append(f"n5_err,{np.abs(p5-truth5).max():.3e}")

    print("benchmark vs twin 5e7-draw references")
    for n in (50, 200):
        mu, V, D = make_problem(n, 2, rng, spread=1.0)
        ta = mc_shares(mu, V, D, 50_000_000, seed=301)
        tb = mc_shares(mu, V, D, 50_000_000, seed=302)
        truth = 0.5 * (ta + tb)
        for R in (1000, 10_000):
            t0 = time.perf_counter()
            p = tilted_shares(mu, V, D, R, seed=7)
            dt = time.perf_counter() - t0
            err = np.abs(p - truth).max()
            print(f"  N={n} R={R}: err {err:.1e}, {dt:.1f}s")
            rows.append(f"N{n}_R{R},{err:.3e};{dt:.2f}s")
        F, W = hermite_nodes(2)
        t0 = time.perf_counter()
        pl = win_probabilities_factor(-mu, V, D, F, W)
        dtl = time.perf_counter() - t0
        print(f"  N={n} lattice: err {np.abs(pl-truth).max():.1e}, {dtl:.2f}s")
        rows.append(f"N{n}_lattice,{np.abs(pl-truth).max():.3e};{dtl:.2f}s")

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
