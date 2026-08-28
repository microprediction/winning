"""Exact N-way Thurstone rating updates.

TrueSkill-style systems propagate skill beliefs through PAIRWISE probit
factors because N-way race factors lacked tractable moments. The shared
survival field supplies those moments exactly:

    E[s_j | i wins]  = m_j + v_j d log p_i / d m_j
    Var[s_j | i wins] = v_j + v_j^2 d^2 log p_i / d m_j^2

and because the choice Jacobian is symmetric, ONE O(QNL) Jacobian-vector
product returns the gradient row for every player simultaneously. Both
identities are verified against brute-force Monte Carlo in the tests.

update_winner: the exact N-way update from winner-only data.
pairwise_update_winner: the classical decomposition (winner beats each
loser in independent two-player probit updates) for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..factor.core import jacobian_vector_product, win_probabilities_factor

_F1 = np.zeros((1, 1))
_W1 = np.ones(1)


def _grad_logp_row(m, D, i):
    """d log p_i / d m_j for all j, via one symmetric-Jacobian JVP."""
    n = len(m)
    Vz = np.zeros((n, 1))
    a = -np.asarray(m, dtype=float)
    p = win_probabilities_factor(a, Vz, D, _F1, _W1)
    e = np.zeros(n)
    e[i] = 1.0
    Ji = jacobian_vector_product(a, Vz, D, _F1, _W1, e, form="grid")
    return -Ji / max(p[i], 1e-300), p[i]


def update_winner(m, v, winner, beta2=1.0, eps=1e-4):
    """Exact-moment posterior (m, v) update given `winner` won the race.

    m, v: prior skill means and variances; beta2: performance noise
    variance. Second derivatives currently by central differences of the
    gradient row (2 extra JVP-row calls per coordinate would be exact; the
    diagonal-only FD used here costs two full rows)."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    D = v + beta2
    g, p_i = _grad_logp_row(m, D, winner)
    m_new = m + v * g
    # diagonal second derivatives by per-coordinate central differences of
    # the gradient row (analytic second-order pass is a known follow-up)
    d2 = np.empty(len(m))
    for j in range(len(m)):
        ej = np.zeros(len(m)); ej[j] = eps
        gp, _ = _grad_logp_row(m + ej, D, winner)
        gm, _ = _grad_logp_row(m - ej, D, winner)
        d2[j] = (gp[j] - gm[j]) / (2 * eps)
    v_new = np.maximum(v + v**2 * d2, 1e-6)
    return m_new, v_new, p_i


def pairwise_update_winner(m, v, winner, beta2=1.0):
    """Classical decomposition: winner beats each loser, independent
    two-player Thurstone (probit) EP updates applied sequentially."""
    m = np.asarray(m, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    for j in range(len(m)):
        if j == winner:
            continue
        c = np.sqrt(v[winner] + v[j] + 2 * beta2)
        t = (m[winner] - m[j]) / c
        phi = np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
        Phi = max(ndtr(t), 1e-300)
        lam = phi / Phi                    # hazard of the win margin
        kappa = lam * (lam + t)
        m[winner] += v[winner] / c * lam
        m[j] -= v[j] / c * lam
        v[winner] *= max(1 - v[winner] / c**2 * kappa, 1e-3)
        v[j] *= max(1 - v[j] / c**2 * kappa, 1e-3)
    return m, v


def update_ranking(m, v, order, beta2=1.0):
    """Full-ranking update: the ranking factorizes exactly as a sequence of
    winner-of-remaining events, P(order) = prod_t P(order[t] wins among
    order[t:]); each stage applies the exact winner update on the shrinking
    field (stage-wise exact moment matching).

    order: indices from first to last finisher.

    HONEST CAVEAT (measured, season_ranked): this sequential decomposition
    treats each stage as a fresh race with fresh noise, but a real
    ranking's stages share ONE performance realization per player.
    Reference TrueSkill respects that shared structure and beats this
    update on full rankings (RMSE 0.085 vs 0.310 at 1500 races), while the
    exact winner update dominates on winner-only data (0.331 vs 0.815).

    The same defect biases STRUCTURE LEARNING, and the bias is measured
    (bandits repo, family-vs-outsider world, 2026-08-28): learning a
    correlation scale by gradient ascent on stagewise-decomposed rankings
    under the Gaussian base inflates it three-fold (s_hat 1.53 +/- 0.36
    against 0.56 +/- 0.05 from winner-only events on the same data) --
    the shared realization across stages masquerades as factor
    correlation. Stagewise is exact under the Gumbel base only (IIA);
    Gaussian-base pipelines should consume winner-only or
    top-1-of-subset events until exact ranked-race moments with shared
    noise exist. That remains the open item."""
    m = np.asarray(m, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    order = list(order)
    for t in range(len(order) - 1):
        rest = np.array(order[t:])
        w_local = 0
        mm, vv, _ = update_winner(m[rest], v[rest], w_local, beta2)
        m[rest], v[rest] = mm, vv
    return m, v


def _order_pass(m, sd, order, L=2001):
    """Joint ordered-statistics likelihood for independent Gaussians and its
    exact gradient, by one forward and one adjoint sweep, O(nL).

    Structure: P = g_1^T D T_2 with T_t = C(g_t * T_{t+1}), T_n = F_n,
    C = trapezoidal cumulative integral, D = dx. P is linear in each
    player's density row g_t, so with the adjoint u_{t+1} = g_t * (C^T u_t)
    every partial derivative is an inner product. Stages carry explicit
    log-scales so 20-player orders (P ~ 1e-19) stay accurate.
    Returns (log P, d log P / d m)."""
    n = len(order)
    lo = float((m - 8 * sd.max()).min())
    hi = float((m + 8 * sd.max()).max())
    x = np.linspace(lo, hi, L)
    dx = x[1] - x[0]

    def cum(y):                        # trapezoidal cumulative integral
        c = np.cumsum(y) * dx
        return c - 0.5 * dx * (y + y[0])

    def cum_T(u):                      # its transpose (reverse form)
        c = np.cumsum(u[::-1])[::-1] * dx
        return c - 0.5 * dx * (u + u[-1])

    g = np.empty((n, L)); dg = np.empty((n, L))
    for t, j in enumerate(order):
        z = (x - m[j]) / sd[j]
        g[t] = np.exp(-0.5 * z * z) / (sd[j] * np.sqrt(2 * np.pi))
        dg[t] = g[t] * z / sd[j]

    # forward sweep: scaled T_t for t = n .. 2
    T = np.empty((n + 1, L)); sT = np.zeros(n + 1)
    j = order[-1]
    T[n] = ndtr((x - m[j]) / sd[j])
    for t in range(n - 1, 1, -1):
        raw = cum(g[t - 1] * T[t + 1])
        mx = raw.max()
        if mx <= 0:
            return -np.inf, np.zeros(len(m))
        T[t] = raw / mx
        sT[t] = sT[t + 1] + np.log(mx)

    raw_p = float(np.sum(g[0] * T[2]) * dx)
    if raw_p <= 0:
        return -np.inf, np.zeros(len(m))
    logP = np.log(raw_p) + sT[2]

    grad = np.zeros(len(m))
    grad[order[0]] = float(np.sum(dg[0] * T[2]) * dx) / raw_p

    # adjoint sweep: u_t aligned with T_t, scaled; u_2 = dx * g_1
    u = g[0] * dx
    su = 0.0
    for t in range(2, n):
        w = cum_T(u)                                   # C^T u_t
        # denominator in matching scales: P = <u_t, T_t> e^{su+sT_t}
        denom = float(np.sum(u * T[t]))
        num = float(np.sum(w * dg[t - 1] * T[t + 1]))
        grad[order[t - 1]] = (num / denom) * np.exp(sT[t + 1] - sT[t])
        u = w * g[t - 1]
        mx = u.max()
        if mx <= 0:
            return logP, grad
        u = u / mx
        su += np.log(mx)
    denom = float(np.sum(u * T[n]))
    grad[order[-1]] = float(np.sum(u * (-g[n - 1]))) / denom
    return logP, grad


def update_ranking_exact(m, v, order, beta2=1.0, eps=1e-3):
    """Exact full-ranking update: means from the analytic gradient of the
    ordered-statistics likelihood, variances from a coarse FD of the
    per-coordinate gradient (bounded below).

    On TrueSkill's home model (independent per-player noise, full order
    observed) this reproduces TrueSkill to ~1e-3 per rating -- their EP is
    essentially exact there. The value is in models TrueSkill cannot
    express: beta2 may be a per-player array (consistent vs erratic
    performers), and the recursion extends to factor-correlated skills."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    sd = np.sqrt(v + np.asarray(beta2, dtype=float))
    _, grad = _order_pass(m, sd, order)
    m_new = m + v * grad
    d2 = np.empty(len(m))
    for j in range(len(m)):
        ej = np.zeros(len(m)); ej[j] = eps
        _, gp = _order_pass(m + ej, sd, order)
        _, gm = _order_pass(m - ej, sd, order)
        d2[j] = (gp[j] - gm[j]) / (2 * eps)
    v_new = np.clip(v + v ** 2 * d2, 1e-4, None)
    return m_new, v_new
