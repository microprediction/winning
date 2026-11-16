"""A zero price is a scratched runner: its dividend is +inf, not NaN (#1009).

dividends_from_prices returned NaN for a zero price, which
prices_from_dividends reads as a MISSING quote and prices at nan_value, so
normalize_dividends revived a scratch at ~0.05% and an all-zero book raised
ZeroDivisionError. +inf maps back to 0, as ability_implied_dividends' does
(#665). NaN and negative prices keep their meaning.
"""
import math
import warnings

import numpy as np
import pytest

from winning.classic.lattice_calibration import (
    dividends_from_prices, normalize_dividends, prices_from_dividends)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from winning.research import StatePricer


def test_zero_price_is_infinite_dividend():
    assert dividends_from_prices([0.5, 0.5, 0.0]) == [2.0, 2.0, math.inf]
    assert dividends_from_prices([0.75, 0.25, 0.0], multiplicity=2) == [
        1.0 / (2 * 0.75), 1.0 / (2 * 0.25), math.inf]


def test_round_trip_keeps_the_zero():
    p = [0.75, 0.25, 0.0]
    assert prices_from_dividends(dividends_from_prices(p)) == p


def test_normalize_dividends_idempotent_for_a_scratch():
    once = normalize_dividends([2, 4, math.inf])
    assert once == [1.5, 3.0, math.inf]
    assert normalize_dividends(once) == once


def test_all_zero_book_is_all_infinite():
    assert dividends_from_prices([0.0, 0.0]) == [math.inf, math.inf]
    assert prices_from_dividends(dividends_from_prices([0.0, 0.0])) == [0.0, 0.0]


def test_missing_and_negative_prices_unchanged():
    # a NaN price poisons the normalisation, as before
    assert all(math.isnan(d) for d in dividends_from_prices([0.5, math.nan, 0.5]))
    d = dividends_from_prices([0.5, -0.1, 0.6])
    assert d[0] == 2.0 and math.isnan(d[1]) and d[2] == 1.0 / 0.6


def test_state_pricer_matches():
    np.testing.assert_array_equal(StatePricer.dividends_from_prices([0.5, 0.5, 0.0]),
                                  [2.0, 2.0, np.inf])
    np.testing.assert_array_equal(StatePricer.dividends_from_prices([0.0, 0.0]), [np.inf, np.inf])
    d = StatePricer.dividends_from_prices([0.5, np.nan, 0.5])
    assert d[0] == 2.0 and np.isnan(d[1]) and d[2] == 2.0
    np.testing.assert_array_equal(
        StatePricer.prices_from_dividends(StatePricer.dividends_from_prices([0.75, 0.25, 0.0])),
        [0.75, 0.25, 0.0])
