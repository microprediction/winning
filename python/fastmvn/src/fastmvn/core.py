"""Fast multivariate normal rectangle probabilities for structured
covariance: the scipy.stats.multivariate_normal.cdf drop-in for the
factor-plus-diagonal slice.

P(a <= X <= b) with X ~ N(mu, V V' + diag(D)): conditional on the
r-dimensional factor the coordinates are independent, so the rectangle
probability is an r-dimensional smooth integral of a product of
univariate normal CDFs. Deterministic, milliseconds at dimensions where
Genz-style quadrature needs seconds to minutes (measured: 4.6 s for one
probability at n = 200 via scipy's integrator vs ~2 ms here).

Port of the R package mvtnormfast (r/mvtnormfast in this repository),
whose measured validation carries over: agreement with mvtnorm inside
its own reported error bound, with Botev's minimax tilting to a few
1e-4 relative at probabilities down to 1e-17 (deep tails via Laplace
recentering), and strict refusal on genuinely dense covariance -- an
inexact factorization must never masquerade as the structured case, so
those calls fall back to scipy.stats.multivariate_normal.cdf unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr, ndtri
from scipy.stats import norm


def _halton_unit(r, n):
    primes = (2, 3, 5, 7, 11, 13)[:r]
    out = np.empty((n, r))
    for c, b in enumerate(primes):
        idx = np.arange(1, n + 1) + 20
        h = np.zeros(n)
        f = 1.0 / b
        i = idx.copy()
        while i.max() > 0:
            h += f * (i % b)
            i //= b
            f /= b
        out[:, c] = h
    return out


def _gh_nodes(r, Q):
    x, w = hermegauss(Q)
    w = w / w.sum()
    grids = np.meshgrid(*([x] * r), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for c in range(r):
        W *= w[np.searchsorted(x, F[:, c])]
    keep = W > 1e-12 * W.max()
    return F[keep], W[keep] / W[keep].sum()


def _nodes_for(V, D):
    r = V.shape[1]
    sharp = float(np.max(np.sqrt((V ** 2).sum(axis=1))
                         / np.sqrt(np.maximum(D, 1e-300))))
    if sharp > 3.0 or r > 2:
        # scrambled Sobol, not Halton: plain Halton's low-dimensional
        # projections degrade badly past three dimensions (measured: a
        # rank-6 auto-detected decomposition gave 1e-4 error on Halton
        # nodes where Sobol reaches quadrature accuracy).
        from scipy.stats import qmc
        n = 2 ** 13
        u = qmc.Sobol(r, scramble=True, seed=0).random(n)
        F = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        return F, np.full(n, 1.0 / n)
    Q = int(np.clip(np.ceil(8.0 * sharp), 15, 201 if r == 1 else 41))
    return _gh_nodes(r, Q)


def factorize_covariance(sigma, max_rank=6, tol=1e-11, n_iter=300):
    """Exact V V' + diag(D) decomposition of sigma, if one exists.

    Iterated principal-factor fit for ranks 1..max_rank, accepted only
    after a recomputed-V verification against the final D; returns
    (V, D) or None.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = len(sigma)
    scale = float(np.abs(sigma).max())
    for r in range(1, min(max_rank, n - 1) + 1):
        D = np.full(n, 0.5 * float(np.mean(np.diag(sigma))))
        for _ in range(n_iter):
            lam, U = np.linalg.eigh(sigma - np.diag(D))
            idx = np.argsort(lam)[::-1][:r]
            V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
            D_new = np.maximum(np.diag(sigma) - (V ** 2).sum(axis=1), 1e-12)
            if np.abs(D_new - D).max() < 1e-12 * scale:
                D = D_new
                break
            D = D_new
        lam, U = np.linalg.eigh(sigma - np.diag(D))
        idx = np.argsort(lam)[::-1][:r]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        if np.abs(V @ V.T + np.diag(D) - sigma).max() < tol * scale:
            return V, D
    return None


def mvn_cdf_fast(lower=None, upper=None, mean=None, sigma=None,
                 V=None, D=None):
    """P(lower <= X <= upper), X ~ N(mean, V V' + diag(D)).

    Supply (V, D), or supply sigma and an exact decomposition is
    searched; dense covariance falls back to
    scipy.stats.multivariate_normal.cdf unchanged. Returns a float with
    .method metadata via the companion function mvn_cdf_fast_info.
    """
    p, _ = _mvn_cdf_impl(lower, upper, mean, sigma, V, D)
    return p


def mvn_cdf_fast_info(lower=None, upper=None, mean=None, sigma=None,
                      V=None, D=None):
    """As mvn_cdf_fast, returning (p, method)."""
    return _mvn_cdf_impl(lower, upper, mean, sigma, V, D)


def _mvn_cdf_impl(lower, upper, mean, sigma, V, D):
    if V is None or D is None:
        if sigma is None:
            raise ValueError("supply sigma, or V and D")
        sigma = np.asarray(sigma, dtype=float)
        fd = factorize_covariance(sigma)
        if fd is None:
            from scipy.stats import multivariate_normal
            n = len(sigma)
            mu = np.zeros(n) if mean is None else np.asarray(mean, float)
            up = np.full(n, np.inf) if upper is None else \
                np.broadcast_to(np.asarray(upper, float), (n,))
            lo = np.full(n, -np.inf) if lower is None else \
                np.broadcast_to(np.asarray(lower, float), (n,))
            mvn = multivariate_normal(mean=mu, cov=sigma,
                                      allow_singular=True)
            p = mvn.cdf(up, lower_limit=lo)
            return float(p), "scipy-fallback"
        V, D = fd
    V = np.atleast_2d(np.asarray(V, dtype=float))
    n = V.shape[0]
    D = np.asarray(D, dtype=float)
    mu = np.zeros(n) if mean is None else np.asarray(mean, dtype=float)
    lo = np.full(n, -np.inf) if lower is None else \
        np.broadcast_to(np.asarray(lower, float), (n,)).astype(float)
    up = np.full(n, np.inf) if upper is None else \
        np.broadcast_to(np.asarray(upper, float), (n,)).astype(float)
    s = np.sqrt(D)

    F, W = _nodes_for(V, D)
    p = _cell_expectation(F, W, V, s, mu, lo, up)
    if p >= 1e-8:
        return float(p), "factor"

    # deep tail: recenter the node set at the Laplace point of the
    # log-integrand and importance-reweight (see r/mvtnormfast).
    r = V.shape[1]

    def logint(f):
        z = V @ f
        cell = ndtr((up - mu - z) / s) - ndtr((lo - mu - z) / s)
        return float(np.log(np.maximum(cell, 1e-300)).sum()
                     - 0.5 * f @ f)

    f0 = np.zeros(r)
    h = 1e-4
    for _ in range(50):
        g = np.array([(logint(f0 + h * e) - logint(f0 - h * e)) / (2 * h)
                      for e in np.eye(r)])
        if np.linalg.norm(g) < 1e-8:
            break
        f0 = f0 + np.clip(0.5 * g, -1, 1)
    from scipy.stats import qmc
    nn = 2 ** 13
    Fh = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=1).random(nn),
                       1e-12, 1 - 1e-12))
    tau = 1.5
    Fq = Fh * tau + f0
    logw = -0.5 * (Fq ** 2).sum(axis=1) + 0.5 * (Fh ** 2).sum(axis=1) \
        + r * np.log(tau)
    M = Fq @ V.T
    hiq = (up - mu)[None, :] - M
    loq = (lo - mu)[None, :] - M
    lc = np.log(np.maximum(ndtr(hiq / s) - ndtr(loq / s), 1e-300))
    lt = lc.sum(axis=1) + logw
    m = lt.max()
    p = float(np.exp(m) * np.mean(np.exp(lt - m)))
    return p, "factor-recentered"


def _cell_expectation(F, W, V, s, mu, lo, up):
    M = F @ V.T
    hi = (up - mu)[None, :] - M
    lo_ = (lo - mu)[None, :] - M
    logcell = np.log(np.maximum(ndtr(hi / s) - ndtr(lo_ / s), 1e-300))
    return float(W @ np.exp(logcell.sum(axis=1)))
