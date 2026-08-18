"""Native contestants. Max-wins convention: p_i = P(U_i is the maximum),
U = mu + V f + sqrt(D) eps. Each method: fn(mu, V, D, budget, seed) ->
(p, info). `budget` is method-specific effort (draws, points, or None for
the method's default resolution); every method documents its own reading.

All ports come from the kinetics research repository (tag paper-r10),
where each carried correctness anchors before entering any comparison.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import qmc

from ..factor.core import (
    hermite_nodes,
    win_probabilities_factor,
)
from .registry import register

try:
    import fastrace as _fastrace
except ImportError:                       # pragma: no cover
    _fastrace = None


def _nodes(k):
    return hermite_nodes(min(k, 4)) if k <= 4 else __import__(
        "winning.factor.core", fromlist=["qmc_nodes"]).qmc_nodes(k)


@register("lattice")
def lattice(mu, V, D, budget=None, seed=None):
    """The shared-survival-field transform (this package's algorithm,
    factor-generalized). budget = lattice points L (default 501)."""
    L = int(budget) if budget else 501
    F, W = _nodes(V.shape[1])
    if _fastrace is not None:
        p, total = _fastrace.win_probabilities_factor(
            -np.asarray(mu, float), np.asarray(V, float),
            np.asarray(D, float), np.ascontiguousarray(F),
            np.ascontiguousarray(W), L)
        return p, {"backend": "fastrace", "L": L, "prenorm_defect": abs(1 - total)}
    p = win_probabilities_factor(-np.asarray(mu, float), V, D, F, W, points=L)
    return p, {"backend": "numpy", "L": L}


@register("direct_mc")
def direct_mc(mu, V, D, budget=100_000, seed=9):
    """Draw utilities, argmax, average. budget = draws."""
    n, k = V.shape
    rng = np.random.default_rng(seed)
    chunk = max(10_000, int(1.5e9 / (n * 32)))
    counts = np.zeros(n)
    done = 0
    while done < budget:
        m = min(chunk, budget - done)
        U = (mu[None, :] + rng.standard_normal((m, k)) @ V.T
             + np.sqrt(D)[None, :] * rng.standard_normal((m, n)))
        counts += np.bincount(np.argmax(U, axis=1), minlength=n)
        done += m
    return counts / counts.sum(), {"draws": budget}


@register("sobol_direct")
def sobol_direct(mu, V, D, budget=2**17, seed=55):
    """Direct simulation with scrambled-Sobol points in dimension k + N."""
    n, k = V.shape
    sob = qmc.Sobol(d=k + n, scramble=True, seed=seed)
    counts = np.zeros(n)
    todo = int(budget)
    while todo > 0:
        blk = min(todo, 2**14)
        Z = ndtri(np.clip(sob.random(blk), 1e-15, 1 - 1e-15))
        U = mu[None, :] + Z[:, :k] @ V.T + np.sqrt(D)[None, :] * Z[:, k:]
        counts += np.bincount(np.argmax(U, axis=1), minlength=n)
        todo -= blk
    return counts / counts.sum(), {"points": int(budget)}


@register("factor_rqmc")
def factor_rqmc(mu, V, D, budget=2**14, seed=11):
    """Per-alternative (k+1)-dimensional conditioned RQMC: the obvious
    fair competitor with the same factor structure, O(R N^2) total."""
    n, k = V.shape
    X = qmc.MultivariateNormalQMC(np.zeros(k + 1), np.eye(k + 1),
                                  seed=seed).random(int(budget))
    f, z = X[:, :k], X[:, k]
    sD = np.sqrt(D)
    p = np.zeros(n)
    for i in range(n):
        others = np.arange(n) != i
        args = (mu[i] - mu[others][None, :] + f @ (V[i] - V[others]).T
                + sD[i] * z[:, None]) / sD[others][None, :]
        p[i] = np.mean(np.prod(ndtr(args), axis=1))
    return p / p.sum(), {"points": int(budget)}


def _ghk_prob(mu, Sigma, i, R, u):
    n = len(mu)
    others = [j for j in range(n) if j != i]
    a = mu[others] - mu[i]
    M = np.zeros((n - 1, n))
    M[np.arange(n - 1), others] = 1.0
    M[:, i] -= 1.0
    C = M @ Sigma @ M.T
    L = np.linalg.cholesky(C + 1e-12 * np.eye(n - 1))
    R = u.shape[0]
    z = np.zeros((R, n - 1))
    logprob = np.zeros(R)
    for t in range(n - 1):
        b = (-a[t] - z[:, :t] @ L[t, :t]) / L[t, t]
        Fb = ndtr(b)
        logprob += np.log(np.maximum(Fb, 1e-300))
        z[:, t] = ndtri(np.clip(u[:, t] * Fb, 1e-300, 1 - 1e-16))
    return float(np.exp(logprob).mean())


@register("ghk")
def ghk(mu, V, D, budget=1000, seed=9):
    """Per-alternative GHK / Genz separation-of-variables, pseudorandom."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    p = np.array([
        _ghk_prob(mu, Sigma, i, budget,
                  np.random.default_rng(seed + i).random((int(budget), n - 1)))
        for i in range(n)])
    return p / p.sum(), {"draws": int(budget)}


@register("qmc_ghk")
def qmc_ghk(mu, V, D, budget=1024, seed=13):
    """GHK with scrambled-Sobol uniforms (Genz-Bretz style)."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    u = qmc.Sobol(d=n - 1, scramble=True, seed=seed).random(int(budget))
    p = np.array([_ghk_prob(mu, Sigma, i, budget, u) for i in range(n)])
    return p / p.sum(), {"draws": int(budget)}


def _tilt_grad(par, L, u):
    n = len(u)
    x = np.zeros(n)
    mu_t = np.zeros(n)
    x[:n - 1] = par[:n - 1]
    mu_t[:n - 1] = par[n - 1:]
    ub = (u - L @ x + np.diag(L) * x) / np.diag(L)
    lnP = log_ndtr(ub - mu_t)
    h = np.exp(-0.5 * (ub - mu_t) ** 2 - 0.5 * np.log(2 * np.pi) - lnP)
    grad_x = x[:n - 1] - mu_t[:n - 1] + h[:n - 1]
    Lr = L / np.diag(L)[:, None]
    cross = (Lr * np.tril(np.ones((n, n)), -1)).T @ h
    return np.concatenate([grad_x, mu_t[:n - 1] + cross[:n - 1]])


@register("tilting")
def tilting(mu, V, D, budget=1000, seed=7):
    """Botev-style minimax exponential tilting, per alternative."""
    from scipy.optimize import root
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    p = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        M = np.zeros((n - 1, n))
        M[np.arange(n - 1), others] = 1.0
        M[:, i] -= 1.0
        Sd = M @ Sigma @ M.T
        m = n - 1
        L = np.linalg.cholesky(Sd + 1e-12 * np.eye(m))
        u = -(mu[np.array(others)] - mu[i])
        sol = root(_tilt_grad, np.zeros(2 * (m - 1)), args=(L, u),
                   method="hybr", options={"maxfev": 4000})
        x = np.zeros(m); mt = np.zeros(m)
        x[:m - 1] = sol.x[:m - 1]
        mt[:m - 1] = sol.x[m - 1:]
        rng = np.random.default_rng(seed + i)
        Z = np.zeros((int(budget), m))
        lw = np.zeros(int(budget))
        for t in range(m):
            ubt = (u[t] - Z[:, :t] @ L[t, :t]) / L[t, t]
            aa = ndtr(ubt - mt[t])
            zt = mt[t] + ndtri(np.clip(rng.random(int(budget)) * aa,
                                       1e-300, 1 - 1e-16))
            Z[:, t] = zt
            lw += log_ndtr(ubt - mt[t]) + 0.5 * mt[t] ** 2 - zt * mt[t]
        mx = lw.max()
        p[i] = np.exp(mx) * np.mean(np.exp(lw - mx))
    return p / p.sum(), {"draws": int(budget)}


@register("stern")
def stern(mu, V, D, budget=1024, seed=0):
    """Stern (1992, Econometrica): smoothed simulation. Split the noise
    into an iid part at scale lam = min(D) and a remainder; draw the
    remainder (factor part plus excess diagonal) by Monte Carlo, and
    conditional on each draw the iid part gives a smooth product-of-CDFs
    integral, here evaluated on a shared lattice per draw. Historical
    implementations evaluated one alternative at a time; the estimator's
    statistical behavior (R^{-1/2} in the remainder draws) is unchanged
    by the shared-field assembly used here."""
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    n = len(mu)
    rng = np.random.default_rng(seed)
    lam = 0.999 * float(D.min())
    sd_ex = np.sqrt(D - lam)
    sqlam = np.sqrt(lam)
    R = int(budget)
    p = np.zeros(n)
    L = 257
    for r in range(R):
        f = rng.standard_normal(V.shape[1])
        eta = V @ f + sd_ex * rng.standard_normal(n)
        a = mu + eta
        lo = a.min() - 8 * sqlam
        hi = a.max() + 8 * sqlam
        x = np.linspace(lo, hi, L)
        dx = x[1] - x[0]
        z = (x[None, :] - a[:, None]) / sqlam
        logF = log_ndtr(z)
        total = logF.sum(axis=0)
        g = np.exp(-0.5 * z * z) / (sqlam * np.sqrt(2 * np.pi))
        w = np.exp(np.clip(total[None, :] - logF, -745.0, 0.0))
        p += (g * w).sum(axis=1) * dx
    p = np.maximum(p / R, 0.0)
    return p / p.sum(), {"R": R, "lam": lam}
