"""Every likelihood door rejects malformed outcomes and scales instead
of scoring them (#129, #366, #390)."""

import numpy as np
import pytest

from winning.factor.permutations import (ordered_probabilities,
                                         plackett_luce_prefix_logprob)
from winning.factor.races import (abilities_from_race, abilities_from_softmax,
                                  plackett_luce_order_logprob,
                                  race_probabilities, softmax_probabilities)
from winning.likelihood import ranking_loglik_and_score
from winning.ratings.full import update_order_full
from winning.ratings.nway import (order_loglik, update_order_correlated,
                                  update_ranking, update_ranking_exact)

MU3 = np.array([0.0, 1.0, 2.0])
V3 = np.array([[0.3], [-0.1], [0.2]])


# --- ranking_loglik_and_score (#390) ---------------------------------

@pytest.mark.parametrize("orders", [[[0, 0]], [[-1]], [[3]], [[0.9]],
                                    [[0, 1, 2, 0]], [[np.nan]]])
def test_ranking_likelihood_rejects_malformed_rows(orders):
    with pytest.raises(ValueError):
        ranking_loglik_and_score(np.zeros((1, 3)), np.zeros((3, 1)), orders)


@pytest.mark.parametrize("orders", [[], [[0], [1]]])
def test_ranking_likelihood_needs_one_row_per_observation(orders):
    with pytest.raises(ValueError, match="one ranking per observation"):
        ranking_loglik_and_score(np.zeros((1, 3)), np.zeros((3, 1)), orders)


def test_valid_rankings_are_unchanged():
    mu = np.zeros((3, 3))
    ll, _, _ = ranking_loglik_and_score(mu, np.zeros((3, 1)),
                                        [[0], [0, 1], [2, 0, 1]])
    assert abs(ll - (np.log(1 / 3) + np.log(1 / 6) + np.log(1 / 6))) < 1e-12


# --- Plackett-Luce orders and prefixes (#129) ------------------------

@pytest.mark.parametrize("kw", [{}, {"V": V3, "F": np.array([[-1.], [1.]]),
                                     "W": np.array([.5, .5])}])
def test_duplicate_prefix_is_rejected(kw):
    with pytest.raises(ValueError, match="repeats"):
        plackett_luce_prefix_logprob(MU3, [0, 0], **kw)


@pytest.mark.parametrize("order", [[0, 0, 1], [0, 1], [0, 1, 3]])
def test_full_order_must_be_a_permutation(order):
    with pytest.raises(ValueError):
        plackett_luce_order_logprob(MU3, order)


# --- the ratings order paths (#129 comment) --------------------------

def test_singleton_partial_order_is_a_tautology():
    m, v = MU3, np.ones(3)
    lp, g = order_loglik(m, np.ones(3), [1])
    assert lp == 0.0 and not g.any()
    mm, vv = update_ranking_exact(m, v, [1])
    assert np.array_equal(mm, m) and np.array_equal(vv, v)
    # exactly, not to the roundoff of a node-weight sum: through the
    # mixture logZ was log(sum W) = 2.2e-16 on Linux CI
    mm, vv, lz = update_order_correlated(m, v, [1], V3)
    assert lz == 0.0 and np.array_equal(mm, m) and np.array_equal(vv, v)
    for Vf in (None, V3):
        mm, S, lz = update_order_full(m, np.eye(3), [1], V=Vf)
        assert lz == 0.0 and np.array_equal(mm, m)
        assert np.array_equal(S, np.eye(3))
    from winning.ratings.teams import update_team_order_full
    A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mm, S, lz = update_team_order_full(m, np.eye(3), A, [1])
    assert lz == 0.0 and np.array_equal(mm, m) and np.array_equal(S, np.eye(3))


@pytest.mark.parametrize("fn", [
    lambda o: order_loglik(MU3, np.ones(3), o),
    lambda o: update_ranking_exact(MU3, np.ones(3), o),
    lambda o: update_ranking(MU3, np.ones(3), o),
    lambda o: update_order_correlated(MU3, np.ones(3), o, V3),
    lambda o: update_order_full(MU3, np.eye(3), o),
])
@pytest.mark.parametrize("order", [[0, 0], [3, 0], [-1, 0]])
def test_ratings_orders_reject_repeats_and_out_of_range(fn, order):
    with pytest.raises(ValueError):
        fn(order)


# --- temperatures (#366) ----------------------------------------------

BAD_LUCE = [0.0, -1.0, np.nan, np.inf, -np.inf]


@pytest.mark.parametrize("tau", BAD_LUCE)
@pytest.mark.parametrize("fn", [
    lambda t: plackett_luce_prefix_logprob(MU3, [0, 1], temperature=t),
    lambda t: plackett_luce_order_logprob(MU3, [0, 1, 2], temperature=t),
    lambda t: plackett_luce_order_logprob(MU3, [0, 1, 2], temperature=t,
                                          V=V3),
    lambda t: softmax_probabilities(MU3, temperature=t),
    lambda t: abilities_from_softmax([0.2, 0.3, 0.5], temperature=t),
    lambda t: ranking_loglik_and_score(MU3[None, :], np.zeros((3, 1)),
                                       [[0, 1]], temperature=t),
])
def test_luce_scale_must_be_finite_and_positive(fn, tau):
    with pytest.raises(ValueError, match="temperature"):
        fn(tau)


@pytest.mark.parametrize("tau", [-1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fn", [
    lambda t: race_probabilities(MU3, temperature=t),
    lambda t: ordered_probabilities(MU3, k=1, temperature=t),
    lambda t: abilities_from_race([0.5, 0.3, 0.2], temperature=t),
])
def test_softening_scale_must_be_finite_and_nonnegative(fn, tau):
    with pytest.raises(ValueError, match="temperature"):
        fn(tau)


def test_zero_is_still_the_hard_race_and_valid_scales_round_trip():
    a = race_probabilities(MU3, temperature=0.0)
    assert np.allclose(a, race_probabilities(MU3))
    p = np.array([0.2, 0.3, 0.5])
    mu = abilities_from_softmax(p, temperature=0.7)
    assert np.allclose(softmax_probabilities(mu, temperature=0.7), p)
    # and the zero-loading score stays exactly zero at a valid scale
    _, _, dV = ranking_loglik_and_score(np.array([[1.0, 0.0]]),
                                        np.zeros((2, 1)), [[0, 1]])
    assert np.abs(dV).max() < 1e-15
