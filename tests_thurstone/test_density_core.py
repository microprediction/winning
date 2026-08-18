import numpy as np
from numpy.testing import assert_allclose


def test_density_is_normalized(base):
    assert_allclose(np.sum(base.p), 1.0, atol=1e-12)


def test_center_tracks_mean(base, grid):
    shifted = base.shift_fractional(+2.5)  # 2.5 lattice steps
    recentered = shifted.center()
    assert abs(recentered.mean()) < 5e-3


def test_integer_and_fractional_shifts(base):
    k = 7
    i = base.shift_fractional(k)
    j = base.shift_integer(k)
    # integer shift == fractional at integer offsets (within tolerance)
    assert np.allclose(i.p, j.p, atol=1e-10)


def test_convolution_preserves_mean(base):
    dsum = base.convolve(base)
    assert abs(dsum.mean() - (base.mean() + base.mean())) < 1e-6


def test_large_shift_does_not_change_length(base):
    K = base.cdf().shape[0]
    from winning.thurstone.density import Density

    # internal integer shift should keep length
    shifted = Density(base.lattice, base.p).shift_integer(K + 100)  # zero-ish
    assert shifted.p.shape[0] == base.p.shape[0]
    s = float(np.sum(shifted.p))
    assert 0.0 <= s <= 1.0 + 1e-12
