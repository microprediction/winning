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
    def inv(p):
        from winning.factor.races import abilities_from_race
        return -abilities_from_race(p)
    mu, var, logZ = update_market(m, v, p_mkt, tau2=tau2, invert=inv)
    # dense reference
    P = np.eye(n) - np.ones((n, n)) / n
    A = np.diag(1.0 / v) + P / tau2
    S = np.linalg.inv(A)
    yc = inv(p_mkt); yc = yc - yc.mean()
    mu_ref = S @ (m / v + yc / tau2)
    assert np.abs(mu - mu_ref).max() < 1e-8
    assert np.abs(var - np.diag(S)).max() < 1e-8
    assert np.isfinite(logZ)
    # gauge subtlety, asserted correctly: with HETEROGENEOUS prior
    # variances the level and contrasts are prior-correlated, so the
    # posterior mean legitimately moves; with UNIFORM v the level is
    # untouched
    v_u = np.full(n, 0.7)
    mu_u, _, _ = update_market(m, v_u, p_mkt, tau2=tau2, invert=inv)
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
    def inv(p):
        from winning.factor.races import abilities_from_race
        return -abilities_from_race(p)
    mu, var, _ = update_market(m, v, p_mkt, tau2=1e-4, invert=inv)
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
    from winning.factor.races import race_probabilities, abilities_from_race
    def inv(p):
        return -abilities_from_race(p)
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
                kw.update(p_market=p_mkt, tau2=tau2, invert=inv)
            if mode in ("both", "outcome"):
                kw.update(winner=winner)
            mh, vh, _ = update_race(m0, v0, **kw)
            err[mode].append(np.mean(((mh - mh.mean()) - (s - s.mean())) ** 2))
    both, price, outcome = (np.mean(err[k]) for k in ("both", "price",
                                                      "outcome"))
    assert both < price < outcome
