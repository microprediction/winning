"""Inversion of top-k memberships: locations at any depth, and the
two-curve (loc, scale) calibration. Exact round trips referee every
solver -- the forward map is the package's own exact quadrature, so
recovery up to gauge is the whole correctness claim; Monte Carlo stays
upstream in test_topk.py where the forward map itself is refereed."""
import numpy as np
import pytest


def test_round_trip_across_depths():
    """Forward-price a seeded field at several depths and invert each
    curve alone: the mean-zero locations must come back."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    rng = np.random.default_rng(7)
    mu = rng.normal(size=9)
    mu -= mu.mean()
    for k in (1, 2, 3, 6):
        q = top_k_probabilities(mu, k)
        back = abilities_from_topk(q, k)
        assert np.abs(np.asarray(back) - mu).max() < 2e-5, f"k={k}"


def test_k1_matches_the_race_inversion():
    """Depth one is the win race: both inverters must agree on the
    mean-zero quotient."""
    from winning.factor.races import abilities_from_race
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu = np.array([-1.2, -0.3, 0.1, 0.4, 1.0])
    q = top_k_probabilities(mu, 1)
    a = np.asarray(abilities_from_topk(q, 1))
    b = np.asarray(abilities_from_race(q, points=513))
    b -= b.mean()
    assert np.abs(a - b).max() < 5e-5


def test_saturated_favorite_converges():
    """A heavy favorite has q -> 1 at k = 2, exactly where log
    residuals die; the logit iteration must still converge."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu = np.array([-3.0, -0.2, 0.3, 0.7, 1.0, 1.2])
    mu -= mu.mean()
    q = top_k_probabilities(mu, 2)
    assert q.max() > 0.97
    back = np.asarray(abilities_from_topk(q, 2))
    assert np.abs(back - mu).max() < 2e-5


def test_factor_round_trip():
    """Rank-one correlation, forward and back through the same node
    mixture."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu = np.array([-0.8, -0.4, 0.0, 0.3, 0.9])
    mu -= mu.mean()
    v = np.array([0.6, 0.6, -0.2, -0.5, -0.5])
    q = top_k_probabilities(mu, 2, V=v)
    back = np.asarray(abilities_from_topk(q, 2, V=v))
    assert np.abs(back - mu).max() < 5e-5


def test_target_contract():
    """Zeros raise, certainty raises after slot renormalization, and
    depth must leave at least one runner out."""
    from winning.factor.topk import abilities_from_topk
    with pytest.raises(ValueError, match="zero top-k"):
        abilities_from_topk([0.5, 0.5, 0.0], 1)
    with pytest.raises(ValueError, match=">= 1"):
        abilities_from_topk([1.2, 0.5, 0.3], 2)
    with pytest.raises(ValueError, match="k must be"):
        abilities_from_topk([0.5, 0.3, 0.2], 3)
    mu, info = abilities_from_topk(
        [0.7, 0.3, 0.0], 1, target_floor=1e-4, return_info=True)
    assert info["floored"].tolist() == [False, False, True]


def test_slopes_are_the_jacobian_diagonal():
    """The inverse iteration's own slopes and the Jacobian's diagonal
    are the same quantity through two different integrals -- the direct
    translation derivative against the cavity cdf, and minus the row
    sum of pair fluxes. Both are exact quadrature; they must agree to
    quadrature precision, for every base that supplies f'."""
    from winning.factor.races import BASES
    from winning.factor.topk import _topk_with_slopes, top_k_jacobian
    mu = np.array([-0.9, -0.2, 0.4, 0.8, 1.1])
    sd = np.array([1.2, 0.8, 1.0, 1.1, 0.9])
    for base in ("normal", "gumbel", "logistic"):
        for k in (1, 2, 3):
            _, sl = _topk_with_slopes(mu, sd, k, BASES[base], 1025)
            J = top_k_jacobian(mu, k, D=sd ** 2, base=base, points=1025)
            assert np.abs(sl - np.diag(J)).max() < 1e-7, (base, k)


def test_round_trip_other_bases():
    """The inversion is base-agnostic wherever the forward map is."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu = np.array([-1.1, -0.5, 0.0, 0.2, 0.5, 0.9])
    mu -= mu.mean()
    for base in ("gumbel", "logistic"):
        q = top_k_probabilities(mu, 2, base=base)
        back = np.asarray(abilities_from_topk(q, 2, base=base))
        assert np.abs(back - mu).max() < 5e-5, base


def test_heteroscedastic_round_trip():
    """Known unequal variances ride through D= unchanged."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    rng = np.random.default_rng(3)
    mu = rng.normal(size=7)
    mu -= mu.mean()
    D = 0.4 + rng.random(7)
    q = top_k_probabilities(mu, 3, D=D)
    back = np.asarray(abilities_from_topk(q, 3, D=D))
    assert np.abs(back - mu).max() < 2e-5


def test_large_field_shallow_and_deep():
    """Twenty runners, shallow and deep memberships, including the
    saturated regime past n/2 where the information lives in the
    longshots."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    rng = np.random.default_rng(19)
    mu = rng.normal(size=20)
    mu -= mu.mean()
    for k in (4, 15):
        q = top_k_probabilities(mu, k)
        back = np.asarray(abilities_from_topk(q, k))
        assert np.abs(back - mu).max() < 5e-5, k


def test_reflection_complement_identity():
    """With a symmetric base, membership of the worst n - k slots is
    the top-(n-k) curve of the reflected field: inverting 1 - q at
    depth n - k must return minus the abilities."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu = np.array([-1.0, -0.3, 0.1, 0.4, 0.8])
    mu -= mu.mean()
    q = top_k_probabilities(mu, 2)
    back = np.asarray(abilities_from_topk(1.0 - q, 3))
    assert np.abs(back + mu).max() < 5e-5


def test_nonconvergence_warns_or_reports():
    """The abilities_from_race contract: starved of iterations, the
    solver warns -- unless the caller asked for the diagnostics."""
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    q = top_k_probabilities(np.array([-1.5, -0.5, 0.5, 1.5]), 2)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        abilities_from_topk(q, 2, n_iter=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, info = abilities_from_topk(q, 2, n_iter=1, return_info=True)
    assert not info["converged"] and info["iterations"] == 1


def test_loc_scale_round_trip():
    """Two curves, 2n - 2 informative numbers, 2n - 2 free parameters:
    the pair must identify both locations and per-runner scales up to
    the two gauges (mean-zero mu, geometric-mean-one sigma)."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_probabilities)
    rng = np.random.default_rng(11)
    n = 8
    mu = rng.normal(size=n)
    sd = np.exp(rng.uniform(-0.3, 0.3, size=n))
    c = np.exp(np.log(sd).mean())
    mu_g = (mu - mu.mean()) / c
    sd_g = sd / c
    q1 = top_k_probabilities(mu, 1, D=sd ** 2)
    q3 = top_k_probabilities(mu, 3, D=sd ** 2)
    mu_h, sd_h = loc_scale_from_topk_pair(q1, 1, q3, 3)
    assert np.abs(np.asarray(mu_h) - mu_g).max() < 5e-4
    assert np.abs(np.asarray(sd_h) - sd_g).max() < 5e-4


def test_loc_scale_separates_spread_from_ability():
    """Equal locations, one erratic runner. At equal mu spread buys
    membership at EVERY depth short of n/2 (reflection symmetry: at
    k = n/2 exactly, a symmetric base makes membership scale-blind,
    q = 1/2 regardless of spread -- the depth this test therefore
    avoids). Ability inflates shallow and deep curves in a different
    ratio than spread does, and the pair solver must attribute the
    pattern to scale, not ability."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_probabilities)
    n = 6
    mu = np.zeros(n)
    sd = np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    q1 = top_k_probabilities(mu, 1, D=sd ** 2)
    q2 = top_k_probabilities(mu, 2, D=sd ** 2)
    q3 = top_k_probabilities(mu, 3, D=sd ** 2)
    assert q1[0] > q1[1]                      # spread buys the win...
    assert q2[0] > q2[1]                      # ...and shallow places
    assert abs(q3[0] - 0.5) < 1e-9            # scale-blind at k = n/2
    mu_h, sd_h, info = loc_scale_from_topk_pair(q1, 1, q2, 2,
                                                return_info=True)
    assert info["converged"]
    assert sd_h[0] > 1.2 * sd_h[1]
    assert np.abs(np.asarray(mu_h)).max() < 5e-3


def test_loc_scale_gumbel_and_field_size():
    """The pair solver at a racing-shaped field: fourteen runners,
    win plus top-3, normal and gumbel bases."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_probabilities)
    rng = np.random.default_rng(23)
    n = 14
    mu = rng.normal(size=n)
    sd = np.exp(rng.uniform(-0.25, 0.25, size=n))
    c = np.exp(np.log(sd).mean())
    mu_g = (mu - mu.mean()) / c
    sd_g = sd / c
    for base in ("normal", "gumbel"):
        q1 = top_k_probabilities(mu, 1, D=sd ** 2, base=base)
        q3 = top_k_probabilities(mu, 3, D=sd ** 2, base=base)
        mu_h, sd_h, info = loc_scale_from_topk_pair(q1, 1, q3, 3, base=base,
                                                    return_info=True)
        assert info["converged"], base
        assert np.abs(np.asarray(mu_h) - mu_g).max() < 1e-3, base
        assert np.abs(np.asarray(sd_h) - sd_g).max() < 1e-3, base


def test_win_and_second_adapter():
    """The market-facing statement: win plus EXACTLY-second marginals
    (columns one and two of the rank marginals) identify (loc, scale)
    through the top-2 conversion."""
    from winning.factor.topk import (loc_scale_from_win_and_second,
                                     rank_probabilities)
    rng = np.random.default_rng(31)
    n = 9
    mu = rng.normal(size=n) * 0.8
    sd = np.exp(rng.uniform(-0.2, 0.2, size=n))
    c = np.exp(np.log(sd).mean())
    mu_g = (mu - mu.mean()) / c
    sd_g = sd / c
    P = rank_probabilities(mu, D=sd ** 2)
    mu_h, sd_h, info = loc_scale_from_win_and_second(P[:, 0], P[:, 1],
                                                     return_info=True)
    assert info["converged"]
    assert np.abs(np.asarray(mu_h) - mu_g).max() < 2e-3
    assert np.abs(np.asarray(sd_h) - sd_g).max() < 2e-3


def test_ridge_and_warm_start():
    """A mild log-sigma ridge (the noisy-board prior) ends converged at
    its penalized optimum, shrinks the sigma dispersion, and biases a
    clean planted field by a measurable, bounded amount -- ridge 0.05
    costs ~0.03 on parameters and ~0.01 max log fit residual at
    sd(log sigma) ~ 0.11, so the prior is not free and stays off by
    default; mu0= is honored."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_probabilities)
    rng = np.random.default_rng(41)
    n = 8
    mu = rng.normal(size=n) * 0.7
    sd = np.exp(rng.uniform(-0.25, 0.25, size=n))
    c = np.exp(np.log(sd).mean())
    mu_g, sd_g = (mu - mu.mean()) / c, sd / c
    q1 = top_k_probabilities(mu, 1, D=sd ** 2)
    q2 = top_k_probabilities(mu, 2, D=sd ** 2)
    mu_h, sd_h, info = loc_scale_from_topk_pair(
        q1, 1, q2, 2, ridge=0.05, mu0=mu_g, return_info=True)
    assert info["converged"]
    assert info["max_logit_residual"] < 0.05
    assert np.abs(np.asarray(mu_h) - mu_g).max() < 5e-2
    assert np.abs(np.asarray(sd_h) - sd_g).max() < 5e-2
    assert np.std(np.log(sd_h)) < np.std(np.log(sd_g)) + 1e-12


def test_rank_marginal_inversion_and_branches():
    """The exactly-second inversion at frozen scales: from a sensible
    warm start it recovers the planted field; from a hostile start it
    may land on ANOTHER field consistent with the same marginal -- the
    two-branch structure the docstrings advertise -- and consistency,
    not recovery, is the honest guarantee there."""
    from winning.factor.topk import (abilities_from_rank_marginal,
                                     rank_probabilities)
    rng = np.random.default_rng(53)
    n = 6
    mu = rng.normal(size=n)
    mu -= mu.mean()
    p2 = rank_probabilities(mu)[:, 1]
    warm = np.asarray(abilities_from_rank_marginal(p2, 2, mu0=mu + 0.05))
    assert np.abs(warm - mu).max() < 1e-4
    cold, info = abilities_from_rank_marginal(p2, 2, mu0=-mu,
                                              return_info=True)
    if info["converged"]:
        achieved = rank_probabilities(np.asarray(cold))[:, 1]
        assert np.abs(np.log(achieved) - np.log(p2)).max() < 1e-6


def test_loc_scale_contract_and_gauge():
    """One depth twice identifies nothing and raises; and a jointly
    rescaled truth prices identically, so the solver's gauged answer
    cannot change."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_probabilities)
    with pytest.raises(ValueError, match="k1 == k2"):
        loc_scale_from_topk_pair([0.6, 0.4], 1, [0.6, 0.4], 1)
    mu = np.array([-0.7, -0.1, 0.2, 0.6])
    sd = np.array([0.9, 1.1, 1.0, 1.0])
    for c in (1.0, 3.0):
        q1 = top_k_probabilities(c * mu, 1, D=(c * sd) ** 2)
        q2 = top_k_probabilities(c * mu, 2, D=(c * sd) ** 2)
        mu_h, sd_h = loc_scale_from_topk_pair(q1, 1, q2, 2)
        if c == 1.0:
            mu_ref, sd_ref = np.asarray(mu_h), np.asarray(sd_h)
        else:
            assert np.abs(np.asarray(mu_h) - mu_ref).max() < 1e-4
            assert np.abs(np.asarray(sd_h) - sd_ref).max() < 1e-4
