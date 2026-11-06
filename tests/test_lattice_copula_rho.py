"""The Gaussian copula's rho is the contestants' correlation, not a loading (#910).

The conditional transform used rho as every contestant's loading on the
common factor, so the realised pairwise correlation was rho**2 and rho and
-rho gave the same prices. Two normal densities have a closed-form win
probability, which pins the contract.
"""
import math

import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density, state_prices_from_densities
from winning.classic.normaldist import normcdf
from winning.classic.scipyinclusion import using_scipy

pytestmark = pytest.mark.skipif(not using_scipy, reason="the copula integrates with scipy")

if using_scipy:
    from winning.classic.lattice_copula import gaussian_copula_conditional_cdf, gaussian_copula_win

L, UNIT = 100, 0.05
LOC, SCALE = 0.8, 1.3


def two_normals():
    return [skew_normal_density(L=L, unit=UNIT, loc=0.0, scale=1.0, a=0.0),
            skew_normal_density(L=L, unit=UNIT, loc=LOC, scale=SCALE, a=0.0)]


def exact_win(rho):
    # lower is better: P(X0 < X1) for N(0,1), N(LOC, SCALE^2) with correlation rho
    return normcdf(LOC / math.sqrt(1.0 + SCALE**2 - 2 * rho * SCALE))


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_two_normals_match_the_bivariate_normal_at_the_stated_correlation(rho):
    got = gaussian_copula_win(two_normals(), rho=rho)[0]
    assert abs(got - exact_win(rho)) < 2e-3, (rho, got, exact_win(rho))
    # and are far from the loading reading, where the correlation was rho**2
    assert abs(got - exact_win(rho**2)) > 1e-2


def test_zero_correlation_is_the_independent_race():
    ds = two_normals()
    np.testing.assert_allclose(gaussian_copula_win(ds, rho=0.0),
                               state_prices_from_densities(ds), atol=2e-3)


@pytest.mark.parametrize("rho", [-0.5, 1.0, 1.5, math.nan, math.inf])
def test_rho_outside_the_unit_interval_is_rejected(rho):
    with pytest.raises(ValueError):
        gaussian_copula_win(two_normals(), rho=rho)
    with pytest.raises(ValueError):
        gaussian_copula_conditional_cdf([0.5], rho=rho, z=0.0)


def test_conditional_cdf_uses_the_square_root_loading():
    # X = sqrt(rho) Z + sqrt(1-rho) eps: P(X <= x | Z=z) = Phi((x - sqrt(rho) z) / sqrt(1-rho))
    rho, z, c = 0.49, 0.6, 0.3
    x = -0.5244005127080407  # Phi^-1(0.3)
    want = normcdf((x - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    got = gaussian_copula_conditional_cdf([c], rho=rho, z=z)[0]
    assert abs(got - want) < 1e-9
