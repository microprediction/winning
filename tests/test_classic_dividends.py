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
    assert all(math.isfinite(a) for a in bad)
    np.testing.assert_allclose(bad, zero)
    # the worthless runner is the slowest, not the fastest
    assert bad[2] > bad[1] > bad[0]
