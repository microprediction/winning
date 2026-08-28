"""Full-covariance belief updates: the ADF diagonal loss, repaired.

Measured motivation (bandits lane): composing a market observation with
an outcome observation, each leg individually near-exact, left the
composed posterior off by 0.14 sd -- the market's contrast observation
induces strong negative cross-correlations that a diagonal belief
projection discards before the outcome update consumes the prior.

The construction: prior correlation IS just more shared factors. With
belief s ~ N(m, Sigma) and Sigma = L L' (Cholesky), the race
performance X = s + V f + noise is a FACTOR race with loadings [L, V]
and idiosyncratic beta2 -- the engine's native form -- so conditional
on the augmented factor vector the race is independent and every
conditional kernel (winner gradient row, ordered-statistics pass)
applies unchanged. The Gaussian-prior moment identities then give the
FULL posterior:

    E[s | E]   = m + Sigma grad_m log P(E)
    Cov[s | E] = Sigma + Sigma hess_m log P(E) Sigma

with the gradient the posterior-node-weighted conditional gradient and
the full Hessian by central differences of the mixture gradient (each
difference returns a whole row). Sigma-in, Sigma-out, logZ returned.
Diagonal wrappers in nway.py remain the cheap default.
"""

from __future__ import annotations

import numpy as np

from .nway import _grad_logp_row, _order_pass


def _psd_repair(S, floor_frac=1e-8):
    S = 0.5 * (S + S.T)
    lam, U = np.linalg.eigh(S)
    floor = floor_frac * max(float(np.trace(S)) / len(S), 1e-12)
    return (U * np.maximum(lam, floor)) @ U.T


def _augmented_nodes(rank, nodes_log2):
    from ..factor.core import qmc_nodes
    return qmc_nodes(rank, m=nodes_log2)


def _mixture_update_full(m, S, V, beta2, node_logp_grad, nodes_log2=12,
                         eps=1e-3):
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    n = len(m)
    L = np.linalg.cholesky(_psd_repair(S))
    if V is None:
        Vaug = L
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if V.shape[0] != n:
            V = V.T
        Vaug = np.hstack([L, V])
    F, W = _augmented_nodes(Vaug.shape[1], nodes_log2)
    logW = np.log(W)
    shifts = F @ Vaug.T

    def mixture(mm):
        logps = np.empty(len(F))
        grads = np.empty((len(F), n))
        for q in range(len(F)):
            lp, g = node_logp_grad(mm + shifts[q])
            logps[q] = lp
            grads[q] = g
        a = logW + logps
        astar = a.max()
        if not np.isfinite(astar):
            return np.zeros(n), -np.inf
        pw = np.exp(a - astar)
        logZ = astar + np.log(pw.sum())
        omega = pw / pw.sum()
        return omega @ grads, logZ

    G, logZ = mixture(m)
    m_new = m + S @ G
    H = np.empty((n, n))
    for j in range(n):
        ej = np.zeros(n); ej[j] = eps
        gp, _ = mixture(m + ej)
        gm, _ = mixture(m - ej)
        H[j] = (gp - gm) / (2 * eps)
    H = 0.5 * (H + H.T)
    S_new = _psd_repair(S + S @ H @ S)
    return m_new, S_new, float(logZ)


def update_winner_full(m, S, winner, V=None, beta2=1.0, nodes_log2=12,
                       eps=1e-3):
    """Winner observation against a full-covariance belief N(m, S)
    (max-wins). V: optional shared performance factors on top of the
    belief correlation; beta2: idiosyncratic performance noise (scalar
    or per-participant). Returns (m_post, S_post, logZ)."""
    n = len(m)
    D = np.broadcast_to(np.asarray(beta2, dtype=float), (n,)).astype(float)

    def node(mm):
        g, p = _grad_logp_row(mm, D, winner)
        return np.log(max(p, 1e-300)), g

    return _mixture_update_full(m, S, V, beta2, node,
                                nodes_log2=nodes_log2, eps=eps)


def update_order_full(m, S, order, V=None, beta2=1.0, nodes_log2=12,
                      eps=1e-3):
    """Full-order observation against a full-covariance belief
    (max-wins, order best-first). Returns (m_post, S_post, logZ);
    near-impossible orders degrade like order_loglik."""
    n = len(m)
    sd = np.sqrt(np.broadcast_to(np.asarray(beta2, dtype=float),
                                 (n,)).astype(float))
    order = np.asarray(order, dtype=int)

    def node(mm):
        return _order_pass(mm, sd, order)

    return _mixture_update_full(m, S, V, beta2, node,
                                nodes_log2=nodes_log2, eps=eps)


def update_market_full(m, S, p_market, tau2=0.25, invert=None,
                       **market_model):
    """Market prices against a full-covariance belief: the conjugate
    case, exact and closed form. Observation y = P s + eta on the
    contrast space (max-wins; the default invert negates the racing
    engine's min-wins abilities). Returns (m_post, S_post, logZ)."""
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    n = len(m)
    if invert is None:
        from ..factor.races import abilities_from_race

        def invert(p):
            return -abilities_from_race(p, **market_model)
    y = np.asarray(invert(np.asarray(p_market, dtype=float)), dtype=float)
    y = y - y.mean()
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (n,)).astype(float)
    P = np.eye(n) - np.ones((n, n)) / n
    Sinv = np.linalg.inv(_psd_repair(S))
    A = Sinv + P @ np.diag(1.0 / tau2) @ P
    S_new = np.linalg.inv(A)
    m_new = S_new @ (Sinv @ m + P @ (y / tau2))
    M = P @ (S + np.diag(tau2)) @ P
    lam, U = np.linalg.eigh(M)
    keep = lam > 1e-12
    r = y - P @ m
    z = U[:, keep].T @ r
    logZ = float(-0.5 * (np.sum(z * z / lam[keep])
                         + np.sum(np.log(lam[keep]))
                         + keep.sum() * np.log(2.0 * np.pi)))
    return m_new, _psd_repair(S_new), logZ
