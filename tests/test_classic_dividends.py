"""Classic dividend conversion boundary policy (#423)."""
import math

import numpy as np
import pytest

from winning.classic.lattice_calibration import (dividend_implied_ability,
                                                 prices_from_dividends)
from winning.classic.std_calibration import centered_std_density
from winning.research.pricing import StatePricer


@pytest.mark.parametrize("d, want", [
    ([2.0, 4.0], [2 / 3, 1 / 3]),
    ([2.0, 4.0, 0.0], [2 / 3, 1 / 3, 0.0]),
    ([2.0, 4.0, -5.0], [2 / 3, 1 / 3, 0.0]),
    ([2.0, 4.0, float("-inf")], [2 / 3, 1 / 3, 0.0]),
    ([2.0, 4.0, float("inf")], [2 / 3, 1 / 3, 0.0]),
    ([float("inf"), float("inf")], [0.0, 0.0]),
    ([0.0, -1.0], [0.0, 0.0]),
])
def test_prices_from_dividends_boundary(d, want):
    got = prices_from_dividends(d)
    np.testing.assert_allclose(got, want, atol=1e-15)
    assert all(p >= 0 for p in got)
    np.testing.assert_allclose(got, StatePricer.prices_from_dividends(d),
                               atol=1e-15)


def test_missing_quote_becomes_nan_value():
    got = prices_from_dividends([2.0, float("nan"), None], nan_value=8.0)
    np.testing.assert_allclose(got, np.array([0.5, 0.125, 0.125]) / 0.75)


def test_negative_dividend_is_not_calibrated_as_a_signed_probability():
    density = centered_std_density(L=500, unit=0.01)
    bad = dividend_implied_ability([2.0, 4.0, -5.0], density)
    zero = dividend_implied_ability([2.0, 4.0, 0.0], density)
    # the worthless runner never wins: ability +inf, not a finite
    # offset that reprices it at a few percent (#589)
    assert all(math.isfinite(a) for a in bad[:2]) and bad[2] == math.inf
    np.testing.assert_allclose(bad, zero)
    # the worthless runner is the slowest, not the fastest
    assert bad[2] > bad[1] > bad[0]


# --- a scratched entrant stays scratched end to end (#589) -------------
#
# prices_from_dividends maps a zero, negative or infinite dividend to an
# exact zero price; the wrapper then handed that zero to the finite
# lattice inverse, which returned an ordinary offset repricing the
# scratched runner at 4.46% (L=15) and 16.3% (L=3).

@pytest.mark.parametrize("L,unit", [(3, 1.0), (15, 0.2), (500, 0.01)])
@pytest.mark.parametrize("scratch", [math.inf, 0.0, -1.0])
def test_scratched_entrant_is_not_reintroduced(L, unit, scratch):
    from winning.classic.lattice import skew_normal_density, state_prices_from_offsets
    from winning.classic.lattice_calibration import ability_implied_state_prices
    d = skew_normal_density(L, unit, a=0)
    a = dividend_implied_ability([2, 4, scratch], d)
    assert a[2] == math.inf and all(math.isfinite(x) for x in a[:2])
    assert ability_implied_state_prices(a, d)[2] == 0
    # the live pair is calibrated as the two-runner race it is
    live = dividend_implied_ability([2, 4], d)
    np.testing.assert_allclose(a[:2], live)
    if L > 3:
        np.testing.assert_allclose(state_prices_from_offsets(d, a[:2]), [2 / 3, 1 / 3], atol=1e-4)


def test_tiny_finite_longshot_is_not_priced_at_four_percent():
    from winning.classic.lattice import skew_normal_density, state_prices_from_offsets
    d = skew_normal_density(15, 0.2, a=0)
    a = dividend_implied_ability([2, 4, 1e12], d)
    assert state_prices_from_offsets(d, a)[2] < 1e-3      # was 0.0446


def test_finite_book_round_trip_unchanged():
    from winning.classic.lattice import skew_normal_density, state_prices_from_offsets
    from winning.classic.lattice_calibration import prices_from_dividends
    d = skew_normal_density(50, 0.1)
    dv = [2.5, 4.0, 6.0, 15.0]
    a = dividend_implied_ability(dv, d)
    assert all(math.isfinite(x) for x in a)
    np.testing.assert_allclose(state_prices_from_offsets(d, a), prices_from_dividends(dv), atol=1e-4)
