"""The classic extended forward must not dilate an in-window field just
because the density's support is wider than half the lattice (#725).

Its hanging test kept a support of width W inside |o| <= L - W. The
standard L=500, unit=0.01 normal has W = 1000, so that interval was
reversed and every ordinary field took the dilating cluster path: the
8:4:2:1 round trip repriced the second share 0.017 off, against 1.6e-6
for the exact primitive the inverse itself uses."""

import warnings

import numpy as np
import pytest

from winning.classic.lattice import (approximate_support_width,
                                     skew_normal_density,
                                     state_prices_from_extended_offsets,
                                     state_prices_from_offsets)
from winning.classic.lattice_calibration import (ability_implied_state_prices,
                                                 state_price_implied_ability)


def test_standard_grid_round_trip_725():
    density = skew_normal_density(L=500, unit=0.01, a=0)
    assert approximate_support_width(density) > 500
    target = np.array([8, 4, 2, 1], dtype=float)
    target /= target.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ability = state_price_implied_ability(target, density)
    back = np.array(ability_implied_state_prices(ability, density))
    assert np.abs(back - target).max() < 1e-5, back      # was 0.0171


@pytest.mark.parametrize("L,unit,scale", [(500, 0.01, 1.0), (500, 0.01, 0.5),
                                          (50, 0.1, 1.0), (15, 0.2, 1.0)])
@pytest.mark.parametrize("spread", [0.3, 1.0, 3.0, 6.0])
def test_in_window_field_is_priced_exactly_when_support_exceeds_L(
        L, unit, scale, spread):
    density = skew_normal_density(L=L, unit=unit, a=0, scale=scale)
    assert approximate_support_width(density) > L
    offsets = list(np.linspace(-spread / 2, spread / 2, 5) * scale / unit)
    if max(abs(o) for o in offsets) >= L - 2:
        pytest.skip("not representable on this lattice")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext = state_prices_from_extended_offsets(density, offsets)
    direct = state_prices_from_offsets(density, offsets)
    assert np.allclose(ext, direct, rtol=0, atol=1e-14), (ext, direct)


def test_genuinely_hanging_field_still_clusters():
    # offsets past the representable range L - 2 still take the cluster
    # path; the direct primitive clamps them and is 0.26 off here
    density = skew_normal_density(L=500, unit=0.01, a=0)
    offsets = [-600.0, -500.0, 500.0, 600.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext = np.array(state_prices_from_extended_offsets(density, offsets))
    # the continuous normal race: two pairs 10 sd apart, each 1 sd apart
    from scipy.stats import norm
    p_fast = norm.cdf(1 / np.sqrt(2))
    assert abs(ext[0] - p_fast) < 1e-4 and abs(ext[1] - (1 - p_fast)) < 1e-4
    assert ext[2] < 1e-12 and ext[3] < 1e-12
