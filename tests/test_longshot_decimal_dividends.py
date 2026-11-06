"""The longshot power law acts on decimal dividends, not bookmaker odds (#918).

``longshot_adjusted_dividends`` raised ``o + 1`` to the power, a net-odds to
decimal conversion the callers had already made. At the neutral exponent of
one it therefore added one to every decimal dividend, moving a 2.0 favourite
to 3.0 while barely touching a 100.0 longshot, and the adjusted book depended
on the overround. The symmetric test could not see it: adding one to equal
dividends cancels in the normalisation.
"""
import math

import numpy as np

from winning.classic.lattice_calibration import normalize_dividends, prices_from_dividends
from winning.classic.lattice_simulation import longshot_adjusted_dividends, skew_normal_place_pricing

BOOK = [2.0, 10.0, 100.0]


def test_exponent_one_is_the_identity_on_an_asymmetric_book():
    np.testing.assert_allclose(longshot_adjusted_dividends(BOOK, longshot_expon=1.0),
                               normalize_dividends(BOOK))


def test_adjustment_depends_only_on_the_normalised_book():
    # removing or doubling the overround must not change the adjusted prices
    for expon in (1.0, 1.17):
        base = prices_from_dividends(longshot_adjusted_dividends(BOOK, expon))
        for scale in (0.5, 2.0):
            scaled = prices_from_dividends(longshot_adjusted_dividends([scale * o for o in BOOK], expon))
            np.testing.assert_allclose(scaled, base, rtol=1e-12)


def test_power_law_is_a_power_of_the_implied_probabilities():
    # q_i proportional to p_i ** expon, which is what the docstring promises
    expon = 1.17
    p = np.array(prices_from_dividends(BOOK))
    q = p ** expon
    np.testing.assert_allclose(prices_from_dividends(longshot_adjusted_dividends(BOOK, expon)),
                               q / q.sum(), rtol=1e-12)


def test_scratches_still_pass_through():
    adj = longshot_adjusted_dividends(BOOK + [math.inf, 0, -1], longshot_expon=1.0)
    assert adj[3:] == [math.inf] * 3
    np.testing.assert_allclose(adj[:3], normalize_dividends(BOOK))


def test_place_pricing_default_follows_the_supplied_book():
    # the win market simulated at the default exponent reproduces the input
    # probabilities, not a book shifted by one
    np.random.seed(12345)
    prices = skew_normal_place_pricing(BOOK, n_samples=20000)
    got = np.array(prices_from_dividends(prices["win"]))
    want = np.array(prices_from_dividends(BOOK))
    shifted = np.array(prices_from_dividends([o + 1.0 for o in BOOK]))
    assert np.abs(got - want).max() < 0.02, (got, want)
    assert np.abs(got - shifted).max() > 0.03, (got, shifted)
