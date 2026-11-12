"""Single-update rating verbs refuse a malformed observation; the ranking
likelihood refuses non-finite utilities; the choice and order label doors
apply the bool and masked-entry rules."""
import numpy as np
import pytest

from winning.ratings.history import update_margins_full
from winning.ratings.teams import update_team_margins_full, update_team_market_full
from winning.ratings.full import update_market_full
from winning.ratings.market import update_market
from winning.likelihood import ranking_loglik_and_score, check_choice
from winning.outcomes import as_order

M, S, V4 = np.array([0.3, -0.1, 0.0, 0.2]), np.eye(4), np.ones(4)
A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float)


@pytest.mark.parametrize("call", [
    lambda x: update_margins_full(M, S, margins=x),
    lambda x: update_margins_full(M, S, scores=x),
    lambda x: update_market(M, V4, x),
    lambda x: update_market_full(M, S, x),
])
@pytest.mark.parametrize("bad", [[3.0], [1.0], [0.0, 1.0, 2.0], [0.0, np.nan, 1.0, 2.0], [0.0, np.inf, 1.0, 2.0]])
def test_individual_updates_refuse_a_malformed_observation(call, bad):
    # a length-1 vector broadcast and, centred, was a dead heat / a uniform
    # market; a NaN gave NaN means beside a finite shrunk covariance
    with pytest.raises(ValueError, match="one entry per runner|not finite"):
        call(bad)


@pytest.mark.parametrize("bad", [[5.0], [0.0, np.nan]])
def test_team_updates_refuse_a_malformed_observation(bad):
    with pytest.raises(ValueError, match="one entry per runner|not finite"):
        update_team_margins_full(M, S, A, margins=bad)
    with pytest.raises(ValueError, match="one entry per runner|not finite"):
        update_team_market_full(M, S, A, [1.0] if len(bad) == 1 else [0.5, np.nan])


def test_well_formed_observations_still_update():
    m, s, lz = update_margins_full(M, S, margins=[0.0, 1.0, 2.0, 0.5])
    assert np.all(np.isfinite(m)) and np.isfinite(lz)
    m, v, lz = update_market(M, V4, [0.4, 0.3, 0.2, 0.1])
    assert np.all(np.isfinite(m))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_ranking_likelihood_refuses_non_finite_utilities(bad):
    mu = np.array([[0.0, bad, 0.5]])
    with pytest.raises(ValueError, match="not finite"):
        ranking_loglik_and_score(mu, np.ones(3) * 0.3, [[0, 1, 2]])


def test_label_doors_apply_the_bool_and_mask_rules():
    with pytest.raises(ValueError, match="boolean"):
        check_choice([True, False, True], 3, 3)
    with pytest.raises(ValueError, match="masked"):
        check_choice(np.ma.array([0, 1, 2], mask=[0, 1, 0]), 3, 3)
    with pytest.raises(ValueError, match="integer"):
        check_choice(np.array(["0", "1", "2"]), 3, 3)
    with pytest.raises(ValueError, match="complex"):
        check_choice(np.array([0, 1, 1], complex), 3, 3)
    assert list(check_choice(np.array([0, 1, 2], dtype=object), 3, 3)) == [0, 1, 2]
    with pytest.raises(ValueError, match="masked"):
        as_order(np.ma.array([0, 1], mask=[0, 1]), 4)
    with pytest.raises(ValueError, match="integer"):
        as_order(np.array([1, 2], complex), 4)
    assert list(as_order(np.ma.array([0, 1], mask=[0, 0]), 4)) == [0, 1]
