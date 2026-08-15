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
