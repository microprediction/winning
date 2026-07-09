"""Deprecated winning 1.x module preserved for import compatibility.

The horse race problem specialized to N(loc, scale) performance, delegating
to thurstone. Signatures and the lower-is-better ability convention match 1.x.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
from thurstone.conventions import NAN_DIVIDEND, STD_L, STD_SCALE, STD_UNIT

from .lattice_calibration import (
    ability_implied_dividends,
    ability_implied_state_prices,
    dividend_implied_ability,
    state_price_implied_ability,
)

warnings.warn(
    "winning.std_calibration is deprecated; use thurstone.AbilityCalibrator "
    "with thurstone.Density.skew_normal(a=0)",
    DeprecationWarning,
    stacklevel=2,
)


def centered_std_density(
    loc: float = 0.0, L: int = STD_L, unit: float = STD_UNIT, scale: float = STD_SCALE
) -> np.ndarray:
    x = unit * np.arange(-L, L + 1)
    p = np.exp(-0.5 * ((x - loc) / scale) ** 2)
    return p / p.sum()


def std_state_price_implied_ability(
    prices, unit: float = STD_UNIT, L: int = STD_L, scale: float = STD_SCALE
) -> List[float]:
    density = centered_std_density(loc=0.0, unit=unit, L=L, scale=scale)
    return state_price_implied_ability(prices=prices, density=density, unit=unit)


def std_dividend_implied_ability(
    dividends,
    nan_value: float = NAN_DIVIDEND,
    L: int = STD_L,
    unit: float = STD_UNIT,
    scale: float = STD_SCALE,
) -> List[float]:
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return dividend_implied_ability(
        dividends=dividends, density=density, nan_value=nan_value, unit=unit
    )


def std_ability_implied_state_prices(
    ability, unit: float = STD_UNIT, L: int = STD_L, scale: float = STD_SCALE
) -> List[float]:
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return ability_implied_state_prices(ability=ability, density=density, unit=unit)


def std_ability_implied_dividends(
    ability,
    unit: float = STD_UNIT,
    L: int = STD_L,
    scale: float = STD_SCALE,
    nan_value: float = NAN_DIVIDEND,
) -> List[float]:
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return ability_implied_dividends(
        ability=ability, density=density, unit=unit, nan_value=nan_value
    )
