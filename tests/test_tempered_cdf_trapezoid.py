"""The tempered race integrates its convolved base with a trapezoid cdf.

The left-Riemann cumulative sum put each runner's cdf half a cell ahead of
its density on the fixed 4001-point curve grid: a first-order bias of
2.7e-4 here, the same at points=4097 because the curve grid does not move.
The reference is a nested scipy quad of the exact normal-plus-Gumbel
convolution (computed once; it takes a minute)."""
import numpy as np
import pytest

from winning.factor.races import race_probabilities

EXACT = [0.46693365, 0.31923892, 0.21382743]   # mu=[0,.5,1], sd 1, tau 1


@pytest.mark.parametrize("points", [None, 4097])
def test_tempered_race_matches_the_exact_convolution(points):
    kw = {} if points is None else {"points": points}
    p = race_probabilities([0.0, 0.5, 1.0], D=np.ones(3), temperature=1.0, **kw)
    np.testing.assert_allclose(p, EXACT, atol=5e-6)    # was 2.7e-4 off
