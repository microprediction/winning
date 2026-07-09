"""The preserved winning 1.x import surface: deprecation + numerical parity.

Gold values computed with the actual winning==1.0.3 PyPI release (the last
published 1.x; the repo's final 1.x tree called itself 1.0.6 but was never
published). The shims delegate to thurstone and must reproduce 1.x numbers.
"""

import warnings

import pytest


def test_legacy_modules_warn_on_import():
    import importlib

    import winning.std_calibration as legacy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(legacy)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_std_calibration_matches_winning_103():
    import numpy as np

    from winning.std_calibration import (
        std_ability_implied_state_prices,
        std_dividend_implied_ability,
    )

    ability = std_dividend_implied_ability([2, 6, np.nan, 3])
    gold = [-0.550785, 0.294618, 2.990068, -0.214328]  # winning==1.0.3
    assert max(abs(a - g) for a, g in zip(ability, gold)) < 1e-3

    roundtrip = std_ability_implied_state_prices(ability)
    gold_prices = [0.500031, 0.166703, 0.0005, 0.333373]
    assert max(abs(p - g) for p, g in zip(roundtrip, gold_prices)) < 1e-3


def test_lattice_calibration_plumbing():
    import numpy as np

    from winning.lattice_calibration import (
        dividends_from_prices,
        normalize,
        prices_from_dividends,
    )

    prices = prices_from_dividends([2.0, 4.0, 4.0])
    assert abs(sum(prices) - 1.0) < 1e-12
    assert prices[0] == pytest.approx(0.5)
    divs = dividends_from_prices(prices)
    assert divs[0] == pytest.approx(2.0)
    assert normalize([1, 1, 2]) == [0.25, 0.25, 0.5]
    assert np.isfinite(prices).all()


def test_skew_calibration_matches_winning_103():
    import numpy as np

    from winning.skew_calibration import (
        skew_ability_implied_state_prices,
        skew_dividend_implied_ability,
        skew_state_price_implied_ability,
    )

    ability = skew_dividend_implied_ability([2.0, 6.0, np.nan, 3.0])
    gold = [-0.514873, 0.274855, 2.775586, -0.200835]  # winning==1.0.3
    assert max(abs(a - g) for a, g in zip(ability, gold)) < 1e-2
    assert ability[2] == max(ability)  # no-bid horse is worst (lower is better)
    prices = skew_ability_implied_state_prices(ability)
    assert abs(prices[0] - 0.5) < 5e-3
    # 1.x positional order is (prices, unit, L, a, scale)
    pos = skew_state_price_implied_ability([0.5, 0.3, 0.2], 0.1, 500, 1.0)
    kw = skew_state_price_implied_ability([0.5, 0.3, 0.2], a=1.0)
    assert max(abs(x - y) for x, y in zip(pos, kw)) < 1e-12
