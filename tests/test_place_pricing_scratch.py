"""A scratched entrant stays scratched through the place simulation (#620)."""
import math

import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density
from winning.classic.lattice_calibration import ability_implied_state_prices, dividend_implied_ability
from winning.classic.lattice_simulation import longshot_adjusted_dividends, skew_normal_place_pricing


def test_issue_repro_scratch_prices_at_zero():
    d = skew_normal_density(500, 0.01, scale=1.0, a=0)
    adj = longshot_adjusted_dividends([2, 4, math.inf], longshot_expon=1.0)
    assert adj[2] == math.inf                         # was NaN -> nan_value 2000
    a = dividend_implied_ability(adj, d)
    assert a[2] == math.inf                           # was 242.28
    assert ability_implied_state_prices(a, d)[2] == 0  # was 0.0005


def test_live_runners_adjust_as_before():
    # the live entries are the reduced book's adjustment
    np.testing.assert_allclose(longshot_adjusted_dividends([2, 4, math.inf, 0, -1]),
                               longshot_adjusted_dividends([2, 4]) + [math.inf] * 3)
    # a missing quote is still a missing quote, not a scratch
    assert math.isfinite(longshot_adjusted_dividends([2, 4, float("nan")])[0])


def test_scratched_runner_never_places():
    np.random.seed(0)
    prices = skew_normal_place_pricing([2, 4, 8, 16, math.inf], n_samples=2000)
    for bet in ("win", "place2", "place3"):
        assert math.isnan(prices[bet][4]), (bet, prices[bet])  # zero counts
    # with five runners a top-4 slot is everyone else's: never the scratch
    assert math.isnan(prices["place4"][4])


@pytest.mark.parametrize("n_live", [1, 2, 3, 4])
def test_scratches_never_fill_places_in_a_short_field(n_live):
    # fewer than four live runners: placegetters_from_performances asked
    # for four places and nsmallest filled the rest with scratches (#627)
    np.random.seed(1)
    book = [2.0, 4.0, 8.0, 16.0][:n_live] + [math.inf] * (5 - n_live)
    prices = skew_normal_place_pricing(book, n_samples=200)
    for k, bet in enumerate(("win", "place2", "place3", "place4"), start=1):
        p = prices[bet]
        assert all(math.isnan(x) for x in p[n_live:]), (bet, p)  # scratches never place
        if k > n_live:
            assert all(math.isnan(x) for x in p), (bet, p)        # market unavailable
        else:
            assert all(x >= 1.0 for x in p[:n_live]), (bet, p)


def test_placegetter_skips_infinite_scores():
    from winning.classic.lattice_simulation import placegetter
    row = [math.inf, 3.0, math.inf, 1.0]
    assert [placegetter(row, k) for k in range(4)] == [3, 1, None, None]
