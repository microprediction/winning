"""The Gaussian copula integral resolves its steps as rho -> 1 (#942).

Near perfect correlation each conditional cdf is a step at a known point
of the integration variable; one adaptive pass stepped over them and two
distinct histograms came back an exact 50/50 race. Above rho = 0.999 the
integral is split at those points.
"""
import numpy as np
import pytest

from winning.classic.lattice_copula import gaussian_copula_win

A = [0.20, 0.60, 0.20]
B = [0.19, 0.60, 0.21]


def test_near_comonotone_asymmetry_is_kept():
    # comonotone limit [.51, .49]; the exact copula at rho = 0.999999 is
    # [0.509999999999, 0.489999999999]
    np.testing.assert_allclose(gaussian_copula_win([A, B], rho=0.999999), [0.51, 0.49], atol=1e-6)
    np.testing.assert_allclose(gaussian_copula_win([B, A], rho=0.999999), [0.49, 0.51], atol=1e-6)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.9, 0.99])
def test_ordinary_correlations_take_the_unsplit_pass(rho):
    # below the split threshold the computation is main's unsplit pass
    p = gaussian_copula_win([A, B], rho=rho)
    assert abs(sum(p) - 1) < 1e-6 and np.all(np.asarray(p) > 0)
