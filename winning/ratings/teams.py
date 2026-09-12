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

from .full import (_mixture_update_full, _order_kernel, _psd_repair,
                   _winner_kernel)


def update_team_winner_full(m, S, A, winner, V=None, beta2=1.0,
                            nodes_log2=10, eps=None):
    """Team `winner` (row index of A) won. Returns (m_post, S_post,
    logZ) over the PLAYER belief (max-wins)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return _mixture_update_full(m, S, V, beta2, None,
                                nodes_log2=nodes_log2, eps=eps, A=A,
                                kernel=_winner_kernel(winner, A.shape[0]))


def update_team_order_full(m, S, A, order, V=None, beta2=1.0,
                           nodes_log2=10, eps=None, base="normal"):
    """Full team finishing order (best first, rows of A)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return _mixture_update_full(m, S, V, beta2, None,
                                nodes_log2=nodes_log2, eps=eps, A=A,
                                kernel=_order_kernel(order, base=base))


def _cardinal_observation(margins=None, scores=None, lengths_scale=1.0,
                          transform=None):
    """The contrast observation y (centred) and the log-Jacobian of an
    optional sub-linear margin transform, shared by the individual and
    team cardinal nodes. margins= (lengths behind the reference, LOWER
    is better, negated) or scores= (HIGHER is better, used as-is after
    scaling). transform: None, a compression scale c for c*asinh(L/c),
    or a callable on raw margins; the Jacobian keeps evidence-based
    tuning of c honest (a more compressive transform shrinks the data
    and would otherwise win spuriously)."""
    if (margins is None) == (scores is None):
        raise ValueError("pass exactly one of margins= or scores=")
    log_jac = 0.0
    if scores is not None:
        y = np.asarray(scores, dtype=float) * float(lengths_scale)
    else:
        Lm = np.asarray(margins, dtype=float)
        if transform is not None:
            if callable(transform):
                Lt = np.asarray(transform(Lm), dtype=float)
                dL = 1e-6 * (1.0 + np.abs(Lm))
                deriv = (np.asarray(transform(Lm + dL), dtype=float) - Lt) / dL
            else:
                c = float(transform)
                Lt = c * np.arcsinh(Lm / c)
                deriv = 1.0 / np.sqrt(1.0 + (Lm / c) ** 2)
            log_jac = float(np.sum(np.log(np.maximum(
                deriv * float(lengths_scale), 1e-300))))
            Lm = Lt
        y = -Lm * float(lengths_scale)
    return y - y.mean(), log_jac


def _lift_contrast_update(m, S, A, y, Cn, log_jac=0.0):
    """Conjugate update of the FULL belief N(m, S) from a contrast
    observation on the observation space: y = P A s + w with
    w ~ N(0, P Cn P), P the centring projection of the k rows of A.
    The Kalman gain is restricted to the observed contrast subspace
    (eigen-truncated, the centring makes the innovation covariance
    singular); the lift S A' updates every entity's mean and every
    cross-covariance, which is what makes the filter a full-state
    filter rather than a sub-block one. Returns (m, S, logZ)."""
    m = np.asarray(m, dtype=float)
    S = _psd_repair(np.asarray(S, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    P = np.eye(k) - np.ones((k, k)) / k
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
                         + keep.sum() * np.log(2.0 * np.pi))) + log_jac
    return m_new, S_new, logZ


def _noise_cov(k, beta2, meas_var=0.0, V=None):
    B = np.broadcast_to(np.asarray(beta2, dtype=float), (k,)).astype(float)
    Cn = np.diag(B + float(meas_var))
    if V is not None:
        Vm = np.atleast_2d(np.asarray(V, dtype=float))
        if Vm.shape[0] != k:
            Vm = Vm.T
        Cn = Cn + Vm @ Vm.T
    return Cn


def update_team_margins_full(m, S, A, margins=None, V=None, beta2=1.0,
                             lengths_scale=1.0, meas_var=0.0, scores=None,
                             transform=None):
    """Conjugate team cardinal update: margins= (lengths behind, LOWER
    is better, negated internally) or scores= (goals / points / negated
    times, HIGHER is better, used as-is after scaling); transform= as in
    history.update_margins_full. Observed team performance contrasts
    y = P_T A s + w, w ~ N(0, P_T (V V' + beta2 I + meas) P_T). Exact;
    returns (m_post, S_post, logZ) over the PLAYER belief. Full
    margins/scores subsume the finishing order -- use one or the other
    per match, never both."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    y, log_jac = _cardinal_observation(margins, scores, lengths_scale, transform)
    Cn = _noise_cov(A.shape[0], beta2, meas_var, V)
    return _lift_contrast_update(m, S, A, y, Cn, log_jac)


def update_team_market_full(m, S, A, p_market, tau2=0.25, invert=None,
                            **market_model):
    """Market prices for the k rows of A (teams, or a race's entrants
    selected out of the full belief) against the FULL belief: the
    conjugate market node lifted through A. invert: prices -> max-wins
    abilities; the default inverts under the racing engine with
    market_model (V=, D=, ...) and negates. Returns (m_post, S_post,
    logZ)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    if invert is None:
        from ..factor.races import abilities_from_race

        def invert(p):
            return -abilities_from_race(p, **market_model)
    y = np.asarray(invert(np.asarray(p_market, dtype=float)), dtype=float)
    y = y - y.mean()
    tau = np.broadcast_to(np.asarray(tau2, dtype=float), (k,)).astype(float)
    return _lift_contrast_update(m, S, A, y, np.diag(tau))
