"""A field that fits the lattice is priced exactly (#1006).

state_prices_from_extended_offsets centred a field on its mean and then
tested |o| <= L - W, which is right only for a support centred on zero.
A field that fits was sent to the dilating cluster path: 7.9 points off
on an atomic support [0, 7], and the python, R and browser ports took
different branches at unit_ratio 2 because the ports test the support
itself. The reference here prices the same density on a padded lattice,
where nothing can fall off.
"""
import math

import numpy as np
import pytest

from winning.classic.lattice import (implied_L, skew_normal_density,
                                     state_prices_from_extended_offsets as ext,
                                     state_prices_from_offsets as exact)


def padded_exact(density, offsets, pad=200):
    L = implied_L(density)
    big = np.zeros(2 * (L + pad) + 1)
    big[pad:pad + 2 * L + 1] = density
    o = np.asarray(offsets, float)
    return np.asarray(exact(big, list(o - math.floor(o.mean()))))


def fits(density, offsets):
    """some integer shift keeps every shifted support on the lattice"""
    L = implied_L(density)
    supp = np.flatnonzero(np.asarray(density) > 1e-12) - L
    W = supp[-1] - supp[0]
    s_lo, s_hi = (supp[0], supp[-1]) if W < L else (0, 0)
    o = np.asarray(offsets, float)
    lo = max(-L - s_lo - math.floor(o.min()), math.ceil(-(L - 2) - o.min()))
    hi = min(L - s_hi - math.ceil(o.max()), math.floor((L - 2) - o.max()))
    return lo <= hi


def test_the_reported_atomic_field_is_exact():
    L = 10
    d = np.zeros(2 * L + 1); d[L] = 0.5; d[L + 7] = 0.5
    off = [6.25, 2, 3.75, 2, 0.75, 0]
    want = [0.013509, 0.09375, 0.03125, 0.09375, 0.283529, 0.484212]
    np.testing.assert_allclose(ext(d, off), padded_exact(d, off), atol=1e-14)
    np.testing.assert_allclose(ext(d, off), want, atol=1e-6)   # was 0.204 on the 0.75 runner


@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_a_field_that_fits_does_not_depend_on_the_ratio(ratio):
    # the ratio only matters on the cluster path, which a fitting field
    # never reaches; at ratio 2 the ports used to disagree here
    d = skew_normal_density(L=15, unit=0.2, scale=1, a=1.5)
    np.testing.assert_allclose(ext(d, [-20, -2, 2], unit_ratio=ratio),
                               [0.99975086, 0.00024442, 4.72e-06], atol=1e-8)


def test_every_fitting_field_is_exact():
    rng = np.random.default_rng(1006)
    checked = 0
    for t in range(240):
        if t % 3 == 0:
            L = int(rng.integers(6, 25)); d = np.zeros(2 * L + 1)
            idx = rng.choice(2 * L + 1, size=int(rng.integers(2, 5)), replace=False)
            d[idx] = rng.random(len(idx)); d /= d.sum()
        elif t % 3 == 1:
            L = int(rng.integers(15, 60))
            d = np.asarray(skew_normal_density(L=L, unit=0.2, scale=float(rng.uniform(.3, 1.5)),
                                               a=float(rng.uniform(-2, 2))))
        else:
            L = int(rng.integers(10, 40))
            d = rng.random(2 * L + 1) * (np.abs(np.arange(-L, L + 1)) < rng.integers(2, L)); d /= d.sum()
        W = np.ptp(np.flatnonzero(d > 1e-12))
        off = list(np.round(rng.uniform(0, max(1, 2 * L - W), int(rng.integers(2, 7))) * 4) / 4)
        if not fits(d, off):
            continue
        checked += 1
        np.testing.assert_allclose(ext(d, off), padded_exact(d, off), atol=1e-12, err_msg=str((L, off)))
    assert checked > 200   # 27 of these 231 were up to 0.092 off before the fix
