"""A scratched entrant stays scratched through the place simulation (#620)."""
import math

import numpy as np

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
