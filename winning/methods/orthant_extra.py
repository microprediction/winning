"""Second wave of contestants: Genz-Bretz canonical RQMC, Mendell-Elston,
EP for orthants, and a basic Ridgway-style SMC. Each computes max-wins
shares by pricing the per-alternative orthant P(d < 0) for the difference
vector d = U_{-i} - U_i, then normalizing. Anchors (N=2 closed form, N=5
vs Monte Carlo) gate admission, as for the first wave.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import qmc

from .registry import register


def _diff_problem(mu, Sigma, i):
    """Means and covariance of d = U_others - U_i; event is d < 0."""
    n = len(mu)
    others = [j for j in range(n) if j != i]
    a = mu[others] - mu[i]
    M = np.zeros((n - 1, n))
    M[np.arange(n - 1), others] = 1.0
    M[:, i] -= 1.0
    C = M @ Sigma @ M.T
    return a, C + 1e-12 * np.eye(n - 1)


def _order_variables(a, C):
    """Genz-style ordering: hardest constraints (smallest marginal
    probability of d_t < 0, i.e. largest a_t/sqrt(C_tt)) first."""
    z = a / np.sqrt(np.diag(C))
    order = np.argsort(-z)
    return order


def _seq_logprob(a, L, u):
    """Sequential separation-of-variables log-weights for uniforms u."""
    R, m = u.shape
    z = np.zeros((R, m))
    logw = np.zeros(R)
    for t in range(m):
        b = (-a[t] - z[:, :t] @ L[t, :t]) / L[t, t]
        Fb = ndtr(b)
        logw += np.log(np.maximum(Fb, 1e-300))
        z[:, t] = ndtri(np.clip(u[:, t] * Fb, 1e-300, 1 - 1e-16))
    return logw


def _genz_orthant(a, C, points, shifts, seed):
    """Canonical Genz-Bretz: ordering + scrambled Sobol + random shifts."""
    m = len(a)
    order = _order_variables(a, C)
    a = a[order]
    C = C[np.ix_(order, order)]
    L = np.linalg.cholesky(C)
    rng = np.random.default_rng(seed)
    ests = []
    for s in range(shifts):
        u = qmc.Sobol(d=m, scramble=True, seed=seed + s).random(points)
        u = (u + rng.random(m)[None, :]) % 1.0
        logw = _seq_logprob(a, L, u)
        mx = logw.max()
        ests.append(np.exp(mx) * np.mean(np.exp(logw - mx)))
    return float(np.mean(ests))


@register("genz_bretz")
def genz_bretz(mu, V, D, budget=1024, seed=13):
    """Canonical Genz-Bretz RQMC: variable reordering, scrambled Sobol,
    random shifts (3). budget = points per shift."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    p = np.zeros(n)
    for i in range(n):
        a, C = _diff_problem(mu, Sigma, i)
        p[i] = _genz_orthant(a, C, int(budget), 3, seed + 1000 * i)
    return p / p.sum(), {"points": int(budget), "shifts": 3}


def _mendell_elston(a, C):
    """Mendell-Elston sequential moment matching for P(d < 0),
    d ~ N(a, C), with hardest-first ordering."""
    m = len(a)
    order = _order_variables(a, C)
    a = a[order].astype(float).copy()
    C = C[np.ix_(order, order)].astype(float).copy()
    logp = 0.0
    for t in range(m):
        s = np.sqrt(C[t, t])
        z = -a[t] / s
        Pt = ndtr(z)
        logp += float(log_ndtr(z))
        if t == m - 1:
            break
        lam = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / max(Pt, 1e-300)
        # moments of d_t | d_t < 0: mean shift and variance reduction
        mean_t = a[t] - s * lam
        # truncated (upper) normal variance: sigma^2 (1 - z*lam - lam^2)
        var_t = C[t, t] * max(1.0 - z * lam - lam * lam, 1e-12)
        r = C[t + 1:, t] / C[t, t]
        a[t + 1:] += r * (mean_t - a[t])
        C_sub = C[t + 1:, t + 1:]
        C[t + 1:, t + 1:] = C_sub - np.outer(r, r) * (C[t, t] - var_t)
    return np.exp(logp)


@register("mendell_elston")
def mendell_elston(mu, V, D, budget=None, seed=None):
    """Mendell-Elston sequential Gaussian moment matching, optimally
    ordered (deterministic; budget ignored)."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    p = np.zeros(n)
    for i in range(n):
        a, C = _diff_problem(mu, Sigma, i)
        p[i] = _mendell_elston(a, C)
    return p / p.sum(), {"deterministic": True}


def _ep_orthant(a, C, sweeps=30, tol=1e-8):
    """EP for P(d < 0), d ~ N(a, C): axis-aligned truncation sites."""
    m = len(a)
    tau = np.zeros(m)
    nu = np.zeros(m)
    Cinv = np.linalg.inv(C)
    for _ in range(sweeps):
        max_delta = 0.0
        Q = Cinv + np.diag(tau)
        S = np.linalg.inv(Q)
        r = S @ (Cinv @ a + nu)
        for t in range(m):
            v_post = S[t, t]
            m_post = r[t]
            v_cav = 1.0 / max(1.0 / v_post - tau[t], 1e-12)
            m_cav = v_cav * (m_post / v_post - nu[t])
            s = np.sqrt(v_cav)
            z = -m_cav / s
            lam = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) \
                / max(ndtr(z), 1e-300)
            m_new = m_cav - s * lam
            v_new = v_cav * max(1.0 - z * lam - lam * lam, 1e-12)
            tau_new = max(1.0 / v_new - 1.0 / v_cav, 1e-12)
            nu_new = m_new / v_new - m_cav / v_cav
            max_delta = max(max_delta, abs(tau_new - tau[t]))
            tau[t], nu[t] = tau_new, nu_new
            Q = Cinv + np.diag(tau)
            S = np.linalg.inv(Q)
            r = S @ (Cinv @ a + nu)
        if max_delta < tol:
            break
    # EP marginal likelihood (normalization of the tilted Gaussian)
    Q = Cinv + np.diag(tau)
    S = np.linalg.inv(Q)
    r = S @ (Cinv @ a + nu)
    logZ = 0.5 * (np.linalg.slogdet(S)[1] - np.linalg.slogdet(C)[1]
                  + r @ Q @ r - a @ Cinv @ a)
    for t in range(m):
        v_post = S[t, t]
        v_cav = 1.0 / max(1.0 / v_post - tau[t], 1e-12)
        m_cav = v_cav * (r[t] / v_post - nu[t])
        z = -m_cav / np.sqrt(v_cav)
        logZ += float(log_ndtr(z)) \
            + 0.5 * np.log1p(tau[t] * v_cav) \
            - (v_cav * nu[t] ** 2 + 2 * m_cav * nu[t]
               - tau[t] * m_cav ** 2) / (2.0 * (1.0 + tau[t] * v_cav))
    return float(np.exp(logZ))


@register("ep_orthant")
def ep_orthant(mu, V, D, budget=None, seed=None):
    """Expectation propagation for each orthant (Cunningham-style;
    deterministic, budget ignored)."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    p = np.zeros(n)
    for i in range(n):
        a, C = _diff_problem(mu, Sigma, i)
        p[i] = _ep_orthant(a, C)
    return p / p.sum(), {"deterministic": True}


@register("smc_orthant")
def smc_orthant(mu, V, D, budget=1000, seed=17):
    """Basic Ridgway-style SMC: GHK proposals with systematic resampling
    when the effective sample size drops below half."""
    n = len(mu)
    Sigma = V @ V.T + np.diag(D)
    rng = np.random.default_rng(seed)
    p = np.zeros(n)
    R = int(budget)
    for i in range(n):
        a, C = _diff_problem(mu, Sigma, i)
        order = _order_variables(a, C)
        a_o = a[order]
        L = np.linalg.cholesky(C[np.ix_(order, order)])
        m = len(a_o)
        z = np.zeros((R, m))
        logw = np.zeros(R)
        logZ = 0.0
        for t in range(m):
            b = (-a_o[t] - z[:, :t] @ L[t, :t]) / L[t, t]
            Fb = np.maximum(ndtr(b), 1e-300)
            logw += np.log(Fb)
            u = rng.random(R)
            z[:, t] = ndtri(np.clip(u * Fb, 1e-300, 1 - 1e-16))
            w = np.exp(logw - logw.max())
            ess = w.sum() ** 2 / (w * w).sum()
            if ess < R / 2 and t < m - 1:
                mx = logw.max()
                logZ += mx + np.log(np.mean(np.exp(logw - mx)))
                probs = w / w.sum()
                idx = rng.choice(R, size=R, p=probs)
                z = z[idx]
                logw = np.zeros(R)
        mx = logw.max()
        logZ += mx + np.log(np.mean(np.exp(logw - mx)))
        p[i] = np.exp(logZ)
    return p / p.sum(), {"draws": R}
