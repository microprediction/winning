"""Deprecated winning 1.x module preserved for import compatibility.

The horse race problem with skew-normal performance, delegating to thurstone.
Signatures, positional argument order and the lower-is-better ability
convention match winning 1.x exactly (verified against winning==1.0.3 output
in tests/test_legacy.py). As in 1.x, the density is the raw skew-normal on the
lattice, not re-centered.
"""

from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np
from thurstone.conventions import ALT_A, ALT_L, ALT_SCALE, ALT_UNIT

from .lattice_calibration import (
    ability_implied_dividends,
    ability_implied_state_prices,
    dividend_implied_ability,
    state_price_implied_ability,
)

warnings.warn(
    "winning.skew_calibration is deprecated; use thurstone.AbilityCalibrator "
    "with thurstone.Density.skew_normal",
    DeprecationWarning,
    stacklevel=2,
)


def skew_normal_density(
    L: int = ALT_L,
    unit: float = ALT_UNIT,
    loc: float = 0.0,
    scale: float = ALT_SCALE,
    a: float = ALT_A,
) -> np.ndarray:
    """Raw (uncentered) skew-normal pdf on the lattice, as in 1.x."""
    x = unit * np.arange(-L, L + 1)
    t = (x - loc) / scale
    p = np.exp(-0.5 * t * t) * (1.0 + np.array([math.erf(a * ti / math.sqrt(2.0)) for ti in t]))
    return p / p.sum()


def skew_state_price_implied_ability(
    prices, unit: float = ALT_UNIT, L: int = ALT_L, a: float = ALT_A, scale: float = ALT_SCALE
) -> List[float]:
    assert scale > 0, "scale parameter should be positive"
    assert unit > 0, "unit parameter should be positive"
    density = skew_normal_density(L=L, unit=unit, loc=0.0, scale=scale, a=a)
    return state_price_implied_ability(prices=prices, density=density, unit=unit)


def skew_dividend_implied_ability(
    dividends,
    unit: float = ALT_UNIT,
    L: int = ALT_L,
    a: float = ALT_A,
    loc=0,
    scale: float = ALT_SCALE,
) -> List[float]:
    density = skew_normal_density(L=L, a=a, loc=loc, scale=scale, unit=unit)
    return dividend_implied_ability(dividends=dividends, density=density, unit=unit)


def skew_ability_implied_state_prices(
    ability, unit: float = ALT_UNIT, L: int = ALT_L, a: float = ALT_A, scale: float = ALT_SCALE
) -> List[float]:
    density = skew_normal_density(L=L, a=a, loc=0.0, scale=scale, unit=unit)
    return ability_implied_state_prices(ability, density=density, unit=unit)


def skew_ability_implied_dividends(
    ability, unit: float = ALT_UNIT, L: int = ALT_L, a: float = ALT_A, scale: float = ALT_SCALE
) -> List[float]:
    density = skew_normal_density(L=L, a=a, loc=0.0, scale=scale, unit=unit)
    return ability_implied_dividends(ability, density=density, unit=unit)
