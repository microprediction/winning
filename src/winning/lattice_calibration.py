"""Deprecated winning 1.x module preserved for import compatibility.

Delegates to thurstone (the re-implementation of the same SIAM-paper
algorithm). Densities are raw pdf arrays of length 2L+1, as in 1.x, and
abilities keep the 1.x convention (lower is better; scale-free unless a
lattice unit is supplied). Deep internals of the old module
(solve_for_implied_offsets and friends) are not preserved — pin
winning==1.0.3 if you need them, or use thurstone.AbilityCalibrator.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
from thurstone import AbilityCalibrator, Density, StatePricer, UniformLattice
from thurstone.conventions import NAN_DIVIDEND

warnings.warn(
    "winning.lattice_calibration is deprecated; use thurstone.AbilityCalibrator",
    DeprecationWarning,
    stacklevel=2,
)


def _as_density(density: Sequence[float], unit: float) -> Density:
    p = np.asarray(density, dtype=float)
    L = (len(p) - 1) // 2
    return Density(UniformLattice(L=L, unit=float(unit)), p)


def prices_from_dividends(dividends, nan_value: float = NAN_DIVIDEND) -> List[float]:
    return [float(x) for x in StatePricer.prices_from_dividends(dividends, nan_value=nan_value)]


def dividends_from_prices(prices, multiplicity: float = 1.0) -> List[float]:
    return [float(x) for x in StatePricer.dividends_from_prices(prices, multiplicity=multiplicity)]


def state_price_implied_ability(prices, density, unit: float = 1.0) -> List[float]:
    cal = AbilityCalibrator(_as_density(density, unit))
    return [float(a) for a in cal.solve_from_prices(list(prices))]


def dividend_implied_ability(
    dividends, density, nan_value: float = NAN_DIVIDEND, unit: float = 1.0
) -> List[float]:
    return state_price_implied_ability(
        prices=prices_from_dividends(dividends, nan_value=nan_value), density=density, unit=unit
    )


def ability_implied_state_prices(ability, density, unit: float = 1.0, max_depth: int = 3):
    del max_depth  # clustering depth is managed internally by thurstone
    cal = AbilityCalibrator(_as_density(density, unit))
    return [float(p) for p in cal.state_prices_from_ability(list(ability))]


def ability_implied_dividends(
    ability, density, unit: float = 1.0, nan_value: float = NAN_DIVIDEND
) -> List[float]:
    # 1.x accepted nan_value but did not forward it: zero-price entrants come
    # back as NaN, and legacy code detects them with np.isnan. Kept verbatim.
    del nan_value
    prices = ability_implied_state_prices(ability=ability, density=density, unit=unit)
    return [safe_inv(p) for p in prices]


def safe_inv(x, nan_value=np.nan):
    try:
        return 1.0 / x
    except ZeroDivisionError:
        return nan_value


def normalize(p) -> List[float]:
    total = sum(p)
    return [pi / total for pi in p]


def normalize_dividends(dividends) -> List[float]:
    return dividends_from_prices(prices_from_dividends(dividends))
