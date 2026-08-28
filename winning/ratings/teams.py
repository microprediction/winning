"""Team races: rider + horse, crews, relay squads, line-ups.

A team's performance is a weighted sum of member abilities plus team
noise: X_team = A s + V f + eps, with A the (teams x players)
assignment or weight matrix (a row per team, weights over its members;
a rider-horse pair is a row with two ones). The full-covariance core
already speaks this language -- with belief S = L L' the team race has
loadings [A L, V], and posteriors lift back to the members through A'.
Members not in today's race are updated exactly as much as their
correlations warrant (nothing, when uncorrelated).

Winner/order observations are the mixture updates; team margins and a
team market are conjugate, as in the individual case.
"""

from __future__ import annotations

import numpy as np

from .full import (_mixture_update_full, _psd_repair)
from .nway import _grad_logp_row, _order_pass


def update_team_winner_full(m, S, A, winner, V=None, beta2=1.0,
                            nodes_log2=12, eps=1e-3):
    """Team `winner` (row index of A) won. Returns (m_post, S_post,
    logZ) over the PLAYER belief (max-wins)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    D = np.broadcast_to(np.asarray(beta2, dtype=float), (k,)).astype(float)

    def node(mo):
        g, p = _grad_logp_row(mo, D, winner)
        return np.log(max(p, 1e-300)), g

    return _mixture_update_full(m, S, V, beta2, node,
                                nodes_log2=nodes_log2, eps=eps, A=A)


def update_team_order_full(m, S, A, order, V=None, beta2=1.0,
                           nodes_log2=12, eps=1e-3):
    """Full team finishing order (best first, rows of A)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    sd = np.sqrt(np.broadcast_to(np.asarray(beta2, dtype=float),
                                 (k,)).astype(float))
    order = np.asarray(order, dtype=int)

    def node(mo):
        return _order_pass(mo, sd, order)

    return _mixture_update_full(m, S, V, beta2, node,
                                nodes_log2=nodes_log2, eps=eps, A=A)


def update_team_margins_full(m, S, A, margins=None, V=None, beta2=1.0,
                             lengths_scale=1.0, meas_var=0.0, scores=None):
    """Conjugate team cardinal update: margins= (lengths behind, LOWER
    is better, negated internally) or scores= (goals / points / negated
    times, HIGHER is better, used as-is after scaling). Observed team
    performance contrasts y = P_T A s + w,
    w ~ N(0, P_T (V V' + beta2 I + meas) P_T). Exact; returns
    (m_post, S_post, logZ). Full margins/scores subsume the finishing
    order -- use one or the other per match, never both."""
    m = np.asarray(m, dtype=float)
    S = _psd_repair(np.asarray(S, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    if (margins is None) == (scores is None):
        raise ValueError("pass exactly one of margins= or scores=")
    if scores is not None:
        y = np.asarray(scores, dtype=float) * float(lengths_scale)
    else:
        y = -np.asarray(margins, dtype=float) * float(lengths_scale)
    y = y - y.mean()
    P = np.eye(k) - np.ones((k, k)) / k
    B = np.broadcast_to(np.asarray(beta2, dtype=float), (k,)).astype(float)
    Cn = np.diag(B + float(meas_var))
    if V is not None:
        Vm = np.atleast_2d(np.asarray(V, dtype=float))
        if Vm.shape[0] != k:
            Vm = Vm.T
        Cn = Cn + Vm @ Vm.T
    PA = P @ A
    M = PA @ S @ PA.T + P @ Cn @ P
    lam, U = np.linalg.eigh(M)
    keep = lam > 1e-10 * max(float(lam.max()), 1e-300)
    Uk = U[:, keep]
    r = y - PA @ m
    z = Uk.T @ r
    K = S @ PA.T @ (Uk * (1.0 / lam[keep])) @ Uk.T
    m_new = m + K @ r
    S_new = _psd_repair(S - K @ (PA @ S))
    logZ = float(-0.5 * (np.sum(z * z / lam[keep])
                         + np.sum(np.log(lam[keep]))
                         + keep.sum() * np.log(2.0 * np.pi)))
    return m_new, S_new, logZ
