"""Classic forward recursion budget (#775). Each rejection was run
against the pre-fix tree, where it was accepted.

ability_implied_state_prices forwarded max_depth unchecked and the
recursion stops only on max_depth == 0, so -1 and True ran as depth 1 and
1.5, nan and inf as depth 2. It is now a whole number >= 0, as in the
browser."""
import math

import numpy as np
import pytest

from winning.classic.lattice import (skew_normal_density,
                                     state_prices_from_extended_offsets)
from winning.classic.lattice_calibration import (ability_implied_dividends,
                                                 ability_implied_state_prices)


FLAT = [1 / 31] * 31
FIELD = [0, 10, 20, 30]
BY_DEPTH = {0: [1.0, 0, 0, 0],
            1: [0.73784245296, 0.223982995514, 0.03767320938, 0.000501342146],
            2: [0.73784245296, 0.193474399289, 0.059003267863, 0.009679879888]}


@pytest.mark.parametrize("bad", [-1, 1.5, True, False, math.nan, math.inf,
                                 -math.inf, np.bool_(True), "2", None, [2]])
def test_invalid_max_depth_is_refused_775(bad):
    with pytest.raises(ValueError, match="max_depth must be a whole number >= 0"):
        ability_implied_state_prices(FIELD, FLAT, max_depth=bad)


@pytest.mark.parametrize("depth", [0, 1, 2, 3, np.int64(2), 2.0])
def test_valid_max_depth_is_unchanged_775(depth):
    got = ability_implied_state_prices(FIELD, FLAT, max_depth=depth)
    want = state_prices_from_extended_offsets(
        FLAT, [float(a) for a in FIELD], max_depth=int(depth))
    assert got == want                                   # bitwise
    if int(depth) in BY_DEPTH:
        assert np.allclose(got, BY_DEPTH[int(depth)], rtol=0, atol=1e-11)


def test_default_max_depth_and_dividends_unchanged_775():
    d = skew_normal_density(L=50, unit=0.1, a=0)
    ab = [-0.4, 0.0, 0.3, 1.2]
    p =ability_implied_state_prices(ab, d, unit=0.1)
    assert p == state_prices_from_extended_offsets(d, [a / 0.1 for a in ab], max_depth=3)
    assert all(math.isfinite(x) for x in ability_implied_dividends(ab, d, unit=0.1))
