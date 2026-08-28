"""Market prices as a conjugate ability observation, and the two-source
race update: prices = thurstone(s + eta) identify contrasts only, so
the update is exact linear-Gaussian on the contrast space."""
import numpy as np

from winning.ratings.market import update_market, update_race


def test_sherman_morrison_matches_dense_posterior():
    rng = np.random.default_rng(0)
    n = 8
    m = rng.normal(size=n)
    v = 0.4 + rng.random(n)
    s_true = m + np.sqrt(v) * rng.normal(size=n)
    tau2 = 0.09
    y = s_true + np.sqrt(tau2) * rng.normal(size=n)
    from winning.factor.races import race_probabilities
    p_mkt = race_probabilities(-(y - y.mean()))     # market prices (max-wins)
    mu, var, logZ = update_market(m, v, p_mkt, tau2=tau2)
    # dense reference
    P = np.eye(n) - np.ones((n, n)) / n
    A = np.diag(1.0 / v) + P / tau2
    S = np.linalg.inv(A)
    from winning.factor.races import abilities_from_race
    yc = -abilities_from_race(p_mkt); yc = yc - yc.mean()
    mu_ref = S @ (m / v + yc / tau2)
    assert np.abs(mu - mu_ref).max() < 1e-8
    assert np.abs(var - np.diag(S)).max() < 1e-8
    assert np.isfinite(logZ)
    # gauge subtlety, asserted correctly: with HETEROGENEOUS prior
    # variances the level and contrasts are prior-correlated, so the
    # posterior mean legitimately moves; with UNIFORM v the level is
    # untouched
    v_u = np.full(n, 0.7)
    mu_u, _, _ = update_market(m, v_u, p_mkt, tau2=tau2)
    assert abs(mu_u.mean() - m.mean()) < 1e-10


def test_market_update_recovers_ability_with_precise_market():
    # a near-noiseless market should pull beliefs onto the market's
    # implied contrasts
    rng = np.random.default_rng(1)
    n = 6
    m = np.zeros(n)
    v = np.ones(n)
    s_true = rng.normal(size=n)
    from winning.factor.races import race_probabilities
    p_mkt = race_probabilities(-s_true)
    mu, var, _ = update_market(m, v, p_mkt, tau2=1e-4)
    sc = s_true - s_true.mean()
    assert np.abs((mu - mu.mean()) - sc).max() < 5e-3
    # prices pin contrasts, never the level: residual variance is
    # exactly the level share v/n
    assert np.abs(var - 1.0 / n).max() < 5e-3


def test_fused_race_update_beats_either_source_alone():
    # simulate many races; the two-source posterior should be closer to
    # truth than price-only or outcome-only on average
    rng = np.random.default_rng(2)
    n, R = 6, 60
    from winning.factor.races import race_probabilities
    err = {"both": [], "price": [], "outcome": []}
    for _ in range(R):
        s = rng.normal(size=n)
        m0, v0 = np.zeros(n), np.ones(n)
        tau2 = 0.25
        y = s + np.sqrt(tau2) * rng.normal(size=n)
        p_mkt = race_probabilities(-y)
        winner = int(np.argmax(s + rng.normal(size=n)))
        for mode in err:
            kw = {}
            if mode in ("both", "price"):
                kw.update(p_market=p_mkt, tau2=tau2)
            if mode in ("both", "outcome"):
                kw.update(winner=winner)
            mh, vh, _ = update_race(m0, v0, **kw)
            err[mode].append(np.mean(((mh - mh.mean()) - (s - s.mean())) ** 2))
    both, price, outcome = (np.mean(err[k]) for k in ("both", "price",
                                                      "outcome"))
    assert both < price < outcome


def test_bare_call_ranks_the_favorite_highest():
    # the seam the bandits lane caught: a bare update_race with only
    # p_market must give the market favorite the HIGHEST posterior mean
    # (max-wins module convention owned by the default invert)
    from winning.factor.races import race_probabilities
    s = np.array([1.2, 0.3, -0.4, -1.1])
    p_mkt = race_probabilities(-s)          # favorite = runner 0
    m, v, _ = update_race(np.zeros(4), np.ones(4), p_market=p_mkt,
                          tau2=0.1)
    assert int(np.argmax(m)) == 0
    assert (np.argsort(-m) == np.argsort(-s)).all()


def test_update_race_market_leg_inverts_under_the_outcome_model():
    # bandits catch: the named V never reached the market leg, which
    # inverted under the independent map. With V given and no explicit
    # market model, the market now inverts under (V, beta2).
    from winning.factor.races import race_probabilities
    import numpy as np
    rng = np.random.default_rng(3)
    n = 5
    V = np.array([[0.9], [0.8], [-0.5], [0.2], [0.0]])
    s = np.array([0.6, 0.1, -0.1, -0.3, -0.3])
    p_mkt = race_probabilities(-s, V=V, D=np.ones(n))
    m_a, v_a, _ = update_race(np.zeros(n), np.ones(n), p_market=p_mkt,
                              tau2=0.05, V=V)
    # inverting under the correct model recovers s's contrasts closely
    from winning.factor.races import abilities_from_race
    y = -abilities_from_race(p_mkt, V=V, D=np.ones(n))
    assert np.abs((y - y.mean()) - (s - s.mean())).max() < 1e-4
    # and the update pulls means toward those contrasts, favorite first
    assert int(np.argmax(m_a)) == 0
