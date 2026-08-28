"""Market-price updates, and the two-source race update.

Peter's model of a typical race: (1) an outcome (winner or order,
format varies), and (2) market prices, which are Thurstone-based
transforms of the market's own NOISY estimate of ability. A race
observation therefore updates beliefs from one or both sources.

The price half is conjugate. If prices = thurstone(s + eta) with
eta ~ N(0, tau2) market noise, inverting the race map recovers
y = P s + eta on the CONTRAST space (P = I - 11'/n: prices carry no
common level, so none is updated -- the identification fact, again).
With diagonal prior N(m, diag v) the posterior precision is
diag(1/v) + P/tau2 = diag(1/v + 1/tau2) - 11'/(n tau2), a rank-one
downdate solved in closed form by Sherman-Morrison; the exact posterior
is full-covariance and the diagonal is reported (the belief state is
diagonal, matching winning.ratings.nway).

The outcome half is update_winner_correlated / update_order_correlated
(or their independent members), applied to the price-updated prior:
prices and outcome are conditionally independent given s, so the
sequential composition is the correct Bayesian order of play.
"""

from __future__ import annotations

import numpy as np

from .nway import (update_order_correlated, update_ranking_exact,
                   update_winner, update_winner_correlated)


def update_market(m, v, p_market, tau2=0.25, invert=None, **market_model):
    """Conjugate belief update from market prices.

    p_market: the market's win probabilities (dividends: pass 1/div,
    renormalized). tau2: variance of the market's ability-estimation
    noise (scalar or per-runner). invert: optional callable p -> a
    for markets priced under a different model than the outcome race;
    the DEFAULT is -abilities_from_race(p, **market_model): the racing
    engine speaks min-wins, this module speaks max-wins skills
    throughout, and the negation is owned here so a bare call ranks the
    favorite highest (a silent convention mix was caught downstream by
    a sign test; if you supply invert=, supply it max-wins).
    Returns (m_post, v_post_diag, logZ) with logZ the contrast-space
    Gaussian evidence of the observation, for weighting market trust.
    """
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(m)
    if invert is None:
        from ..factor.races import abilities_from_race

        def invert(p):
            return -abilities_from_race(p, **market_model)
    y = np.asarray(invert(np.asarray(p_market, dtype=float)), dtype=float)
    y = y - y.mean()
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (n,)).copy()

    # posterior covariance: (diag(1/v + 1/tau2) - (1/n) q q'/scale)^{-1}
    # via Sherman-Morrison, with the general per-runner tau2 handled by
    # the exact rank-one structure of P'diag(1/tau2)P only when tau2 is
    # uniform; otherwise fall back to the dense solve (n is small in
    # ratings use).
    if np.allclose(tau2, tau2[0]):
        t2 = float(tau2[0])
        c = 1.0 / v + 1.0 / t2
        u = 1.0 / c
        k = (1.0 / (n * t2)) / (1.0 - (1.0 / (n * t2)) * u.sum())
        b = m / v + y / t2
        mu = b * u + k * u * float(u @ b)
        var = u + k * u * u
    else:
        P = np.eye(n) - np.ones((n, n)) / n
        A = np.diag(1.0 / v) + P @ np.diag(1.0 / tau2) @ P
        S = np.linalg.inv(A)
        mu = S @ (m / v + P @ (y / tau2))
        var = np.diag(S).copy()

    # evidence on the contrast space: y ~ N(Pm, P(diag(v)+diag(tau2))P)
    P = np.eye(n) - np.ones((n, n)) / n
    M = P @ np.diag(v + tau2) @ P
    lam, U = np.linalg.eigh(M)
    keep = lam > 1e-12
    r = y - P @ m
    z = U[:, keep].T @ r
    logZ = float(-0.5 * (np.sum(z * z / lam[keep]) + np.sum(np.log(lam[keep]))
                         + keep.sum() * np.log(2.0 * np.pi)))
    return mu, np.maximum(var, 1e-6), logZ


def update_race(m, v, winner=None, order=None, p_market=None, tau2=0.25,
                V=None, beta2=1.0, Qf=7, **market_model):
    """The typical race: update from market prices, an outcome, or both
    (prices first -- the market's information is pre-race -- then the
    outcome on the updated prior). Any of winner/order/p_market may be
    omitted. Returns (m_post, v_post, info) with info holding the logZ
    of each applied source."""
    m = np.asarray(m, dtype=float).copy()
    v = np.asarray(v, dtype=float).copy()
    info = {}
    if p_market is not None:
        if V is not None and not market_model:
            # the seam the bandits integration caught: the named V never
            # reached **market_model, so the market leg inverted under
            # the INDEPENDENT map while the outcome leg used the
            # correlated one. Default the market's pricing model to the
            # outcome model (loadings V, idio beta2); pass market_model
            # kwargs or invert= to price the market differently.
            from ..factor.races import abilities_from_race
            Vm = np.atleast_2d(np.asarray(V, dtype=float))
            if Vm.shape[0] != len(m):
                Vm = Vm.T
            Dm = np.broadcast_to(np.asarray(beta2, dtype=float),
                                 (len(m),)).astype(float)
            market_model = {"invert":
                            lambda p: -abilities_from_race(p, V=Vm, D=Dm)}
        m, v, lz = update_market(m, v, p_market, tau2=tau2, **market_model)
        info["logZ_market"] = lz
    if order is not None:
        if V is not None:
            m, v, lz = update_order_correlated(m, v, order, V, beta2=beta2,
                                               Qf=Qf)
            info["logZ_outcome"] = lz
        else:
            m, v = update_ranking_exact(m, v, order, beta2=beta2)
    elif winner is not None:
        if V is not None:
            m, v, lz = update_winner_correlated(m, v, winner, V,
                                                beta2=beta2, Qf=Qf)
            info["logZ_outcome"] = lz
        else:
            m, v, p = update_winner(m, v, winner, beta2=beta2)
            info["logZ_outcome"] = float(np.log(max(p, 1e-300)))
    return m, v, info
