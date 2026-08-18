import numpy as np
import pytest

from winning.thurstone import Density, UniformLattice

ATOL = 1e-8
RTOL = 1e-8


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def grid():
    return UniformLattice(L=500, unit=0.1)


@pytest.fixture(scope="session")
def base(grid):
    # symmetric base density
    return Density.skew_normal(grid, loc=0.0, scale=1.0, a=0.0)
