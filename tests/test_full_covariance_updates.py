"""Full-covariance belief updates: Sigma in, Sigma out. The referee is
importance-sampled / rejection-sampled MC posteriors including CROSS
covariances, plus the measured fusion scenario that motivated them
(diagonal ADF lost 0.14 sd on market+outcome composition)."""
import numpy as np

from winning.ratings.full import (update_market_full, update_order_full,
                                  update_winner_full)
from winning.ratings.market import update_market
from winning.ratings.nway import update_winner_correlated


def test_winner_full_matches_mc_posterior_including_cross_terms():
    rng = np.random.default_rng(0)
    n = 5
    m = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
    A = rng.normal(size=(n, n)) * 0.3
    S = A @ A.T + 0.5 * np.eye(n)
    V = np.array([[0.8], [0.7], [0.1], [-0.6], [0.2]])
    M = 2_000_000
    s = m + rng.standard_normal((M, n)) @ np.linalg.cholesky(S).T
    X = s + rng.standard_normal((M, 1)) @ V.T + rng.standard_normal((M, n))
    keep = X.argmax(axis=1) == 3
    s_w = s[keep]
    m_mc = s_w.mean(axis=0)
    S_mc = np.cov(s_w.T)
    m_hat, S_hat, logZ = update_winner_full(m, S, 3, V=V, beta2=1.0)
    se = np.sqrt(np.diag(S_mc) / keep.sum())
    assert np.abs(m_hat - m_mc).max() < 5 * se.max() + 5e-3
    assert np.abs(S_hat - S_mc).max() < 0.02          # cross terms included
    assert abs(np.exp(logZ) - keep.mean()) < 5e-3


def test_market_full_is_exact_and_reduces_to_diagonal():
    rng = np.random.default_rng(1)
    n = 6
    m = rng.normal(size=n)
    v = 0.5 + rng.random(n)
    from winning.factor.races import race_probabilities
    y = rng.normal(size=n)
    p_mkt = race_probabilities(-y)
    m_d, v_d, lz_d = update_market(m, v, p_mkt, tau2=0.2)
    m_f, S_f, lz_f = update_market_full(m, np.diag(v), p_mkt, tau2=0.2)
    assert np.abs(m_f - m_d).max() < 1e-9
    assert np.abs(np.diag(S_f) - v_d).max() < 1e-9
    assert abs(lz_f - lz_d) < 1e-9
    # and the off-diagonals are the point: the observation pins
    # CONTRASTS, so what survives is shared level uncertainty -- a
    # near-rank-one POSITIVE block ~ (v/n) 11' that the diagonal
    # projection throws away (measured +0.13 here)
    off = S_f[~np.eye(n, dtype=bool)]
    assert off.mean() > 0.05


def test_fusion_gap_closes_with_full_covariance():
    # the measured motivation: market observation then outcome. The
    # diagonal path discards the market-induced cross-correlations; the
    # full path keeps them. Referee: importance-weighted MC posterior
    # (weight by the market likelihood, condition on the winner).
    rng = np.random.default_rng(2)
    n = 5
    m0 = np.zeros(n)
    v0 = np.ones(n)
    s_true = np.array([0.8, 0.2, 0.0, -0.3, -0.7])
    tau2 = 0.09
    from winning.factor.races import race_probabilities
    y_obs = s_true + np.sqrt(tau2) * rng.standard_normal(n)
    p_mkt = race_probabilities(-y_obs)
    winner = 1                                     # mild upset
    M = 4_000_000
    s = rng.standard_normal((M, n))
    X = s + rng.standard_normal((M, n))
    keep = X.argmax(axis=1) == winner
    s_k = s[keep]
    yc = y_obs - y_obs.mean()
    Pc = s_k - s_k.mean(axis=1, keepdims=True)
    logw = -0.5 * ((Pc - yc) ** 2).sum(axis=1) / tau2
    w = np.exp(logw - logw.max())
    w /= w.sum()
    m_mc = w @ s_k
    ess = 1.0 / np.sum(w ** 2)
    assert ess > 3000
    # diagonal path
    md, vd, _ = update_market(m0, v0, p_mkt, tau2=tau2)
    md2, vd2, _ = update_winner_correlated(md, vd, winner,
                                           np.zeros((n, 1)), beta2=1.0)
    # full path
    mf, Sf, _ = update_market_full(m0, np.diag(v0), p_mkt, tau2=tau2)
    mf2, Sf2, _ = update_winner_full(mf, Sf, winner, beta2=1.0)
    gap_diag = np.abs(md2 - m_mc).max()
    gap_full = np.abs(mf2 - m_mc).max()
    assert gap_full < 0.05
    assert gap_full < 0.6 * gap_diag


def test_diffuse_prior_updates_stay_exact():
    # corner-space audit (bandits): the belief used to ride quadrature,
    # so posterior variance inflated with prior_sd/sqrt(beta2) -- 7x at
    # ratio 100 on the winner path, a hard crash past ratio 20 on the
    # order path. With the belief's diagonal part on the LATTICE both
    # paths track the diagonal members (themselves MC-exact) at every
    # ratio.
    from winning.ratings.full import update_order_full
    from winning.ratings.nway import update_winner_correlated
    n = 3
    for sd in (3.0, 10.0, 30.0, 100.0):
        m0 = np.zeros(n)
        S0 = np.eye(n) * sd * sd
        mf, Sf, _ = update_winner_full(m0, S0, 0, V=np.zeros((n, 1)),
                                       beta2=1.0)
        md, vd, _ = update_winner_correlated(m0, np.full(n, sd * sd), 0,
                                             np.zeros((n, 1)), beta2=1.0)
        assert np.abs(mf - md).max() < 1e-6 * max(sd, 1.0)
        assert np.abs(np.diag(Sf) - vd).max() < 3e-3 * vd.max()
        mo, So, lz = update_order_full(m0, S0, np.array([0, 1, 2]),
                                       V=np.zeros((n, 1)), beta2=1.0)
        assert np.isfinite(mo).all() and np.isfinite(So).all()
        assert np.isfinite(lz)
        # the order arm agrees with the winner arm on the WINNER's
        # coordinate only -- it additionally ranks 2nd against 3rd,
        # which a winner observation is silent about (order splits the
        # losers +-2.41 where winner leaves both at -1.20; both right)
        assert abs(mo[0] - md[0]) < 3e-3 * max(sd, 1.0)
        assert mo[1] > mo[2]
        assert abs(mo.sum() - md.sum()) < 1e-6 * max(sd, 1.0)


def test_relabelling_equivariance():
    # the belief split is eigendecomposition-based, so relabelling the
    # field permutes the answer; residual error is finite-node only and
    # falls with the node budget (2.9e-2 at 2^10 -> 7e-4 at 2^14)
    from winning.ratings.full import _belief_split
    rng = np.random.default_rng(3)
    n = 5
    A = rng.normal(size=(n, n)) * 0.4
    S = A @ A.T + 0.6 * np.eye(n)
    perm = np.array([2, 0, 4, 1, 3])
    B1, psi1 = _belief_split(S)
    B2, psi2 = _belief_split(S[np.ix_(perm, perm)])
    assert np.abs(psi2 - psi1[perm]).max() < 1e-10
    assert np.abs(B2 @ B2.T - (B1 @ B1.T)[np.ix_(perm, perm)]).max() < 1e-10
    m = rng.normal(size=n) * 0.5
    V = rng.normal(size=(n, 2)) * 0.5
    inv = np.argsort(perm)
    m1, S1, _ = update_winner_full(m, S, 1, V=V, beta2=1.0)
    m2, S2, _ = update_winner_full(m[perm], S[np.ix_(perm, perm)],
                                   int(inv[1]), V=V[perm], beta2=1.0)
    assert np.abs(m2 - m1[perm]).max() < 5e-3
