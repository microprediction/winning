"""Round-3 port regressions on the Python side of the ports: the compiled
fastrace boundary (#103, #339) and the classic normalizers (#339)."""
import warnings

import numpy as np
import pytest

from winning.classic.lattice import (as_classic_density, as_classic_prices,
                                     state_prices_from_offsets)
from winning.classic.lattice_calibration import solve_for_implied_offsets


def test_classic_normalizers_are_scale_safe():
    # finite atoms whose raw total overflows were normalised to zeros and
    # priced as a tie (#339)
    d = [0, 0, 1, 1, 0, 0, 0]
    huge = [0, 0, 1e308, 1e308, 0, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(as_classic_density(huge),
                                   as_classic_density(d), rtol=0, atol=0)
        np.testing.assert_allclose(state_prices_from_offsets(huge, [0, 1]),
                                   state_prices_from_offsets(d, [0, 1]),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(as_classic_prices([1e308, 9e307]),
                                   as_classic_prices([1.0, 0.9]),
                                   rtol=1e-15, atol=0)
    a = solve_for_implied_offsets([1e308, 9e307], d, nIter=10)
    b = solve_for_implied_offsets([1.0, 0.9], d, nIter=10)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert abs(a[0] - a[1]) > 1e-3          # not the tie it used to be
    # ordinary input is untouched, bit for bit
    raw = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]) * 3.0
    assert np.array_equal(as_classic_density(raw), raw / raw.sum())


try:
    import fastrace
except ImportError:                     # pure install: compiled tests skip
    fastrace = None
needs_fastrace = pytest.mark.skipif(fastrace is None,
                                    reason="fastrace not installed")


@needs_fastrace
@pytest.mark.parametrize("beta", [0.5, 0.7, 0.999])
def test_fastrace_refuses_exponential_power_below_one(beta):
    # the factory refuses beta < 1 (#103); the compiled export did not
    from scipy.special import gamma
    a = np.sqrt(gamma(1 / beta) / gamma(3 / beta))
    with pytest.raises(ValueError, match="domain"):
        fastrace.forward_and_slopes_base(
            np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
            np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 4, [beta, a])


@needs_fastrace
def test_fastrace_exponential_power_one_still_prices():
    from winning.factor.races import exponential_power_base
    b = exponential_power_base(1.0)
    beta, a = b.rust_base[1]
    p, _, _ = fastrace.forward_and_slopes_base(
        np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
        np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 4, [beta, a])
    assert np.all(np.isfinite(p))


@needs_fastrace
def test_fastrace_student_needs_unit_variance_nu():
    with pytest.raises(ValueError, match="domain"):
        fastrace.forward_and_slopes_base(
            np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
            np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 5, [1.5, 1.0])


@needs_fastrace
def test_fastrace_classic_density_needs_positive_mass():
    # seven zero atoms priced every runner at 0 (#339)
    with pytest.raises(ValueError, match="positive mass"):
        fastrace.classic_exact_state_prices([0.0] * 7, [0.0, 0.0])
    with pytest.raises(ValueError, match="positive mass"):
        fastrace.classic_exact_calibrate([0.0] * 7, [0.5, 0.5],
                                         [1.0, 0.0, -1.0], [0.0, 0.0], 3)
