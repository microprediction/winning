"""Custom-base special functions in the compiled kernel (#134, #193):
enabling Rust must not change a legal custom-base model, including at
extreme but valid shape parameters. Skipped without fastrace."""
import numpy as np
import pytest

fastrace = pytest.importorskip("fastrace")

import winning
from winning.factor.races import (race_probabilities, exponential_power_base,
                                  student_base, skew_normal_base)

F1 = np.zeros((1, 1))
W1 = np.ones(1)


def _both(mu, D, base, points):
    V = np.zeros((len(mu), 1))
    try:
        winning.use_rust(True)
        pr = race_probabilities(mu, V=V, D=D, F=F1, W=W1, base=base,
                                points=points)
        winning.use_rust(False)
        pp = race_probabilities(mu, V=V, D=D, F=F1, W=W1, base=base,
                                points=points)
    finally:
        winning.use_rust(True)
    return pr, pp


def test_exponential_power_large_beta_matches_numpy():
    # the #134 reproducer: was max error 0.0719 at every resolution
    mu = np.array([-1.4, -.5, .2])
    D = np.array([.25, 4., 6.])
    pr, pp = _both(mu, D, exponential_power_base(50), 10001)
    assert np.max(np.abs(pr - pp)) < 1e-9, (pr, pp)


@pytest.mark.parametrize("beta", [0.5, 1.5, 4.0, 12.0, 25.0])
def test_exponential_power_range_matches_numpy(beta):
    mu = np.array([-1.0, -.3, .2, .9])
    D = np.array([.3, 1., 2., 5.])
    pr, pp = _both(mu, D, exponential_power_base(beta), 4001)
    assert np.max(np.abs(pr - pp)) < 1e-9, (beta, pr, pp)


@pytest.mark.parametrize("nu", [2.5, 5.0, 30.0, 1e3, 1e5, 1e8, 1e50])
def test_student_matches_numpy_at_every_nu(nu):
    # large nu used to drift from the normal limit and collapse to a
    # uniform race at nu = 1e50
    mu = np.array([-1.0, -.3, .2, .9])
    D = np.array([.3, 1., 2., 5.])
    pr, pp = _both(mu, D, student_base(nu), 4001)
    assert np.max(np.abs(pr - pp)) < 1e-9, (nu, pr, pp)
    if nu >= 1e8:
        pn, _ = _both(mu, D, "normal", 4001)
        assert np.max(np.abs(pr - pn)) < 1e-7


@pytest.mark.parametrize("alpha", [-10.0, -5.0, -2.0, 3.0])
def test_skew_normal_heterogeneous_scale_matches_numpy(alpha):
    # negative shape: S = Phi(-x) + 2T(x, alpha) cancelled in the right
    # tail, which heterogeneous D makes the small shares depend on (#193)
    mu = np.array([-0.6, 0.0, 0.4, 1.2])
    D = np.array([4.0, 1.0, 0.2, 0.05])
    pr, pp = _both(mu, D, skew_normal_base(alpha), 4001)
    rel = np.abs(pr - pp) / pp
    # SciPy's own skewnorm.sf is ~6e-7 relative off in this tail
    assert np.max(rel) < 1e-5, (alpha, pr, pp)


def test_non_finite_base_parameters_are_value_errors():
    with pytest.raises(ValueError, match="finite nu"):
        student_base(np.inf)
    with pytest.raises(ValueError, match="finite beta"):
        exponential_power_base(np.inf)
    mu = np.zeros(3)
    args = (mu, np.zeros((3, 1)), np.ones(3), F1, W1, 129, -12., 12.)
    for bid, prm in [(5, [np.inf, 1.0]), (4, [np.nan, 1.0]),
                     (6, [1.0, 0.0, -1.0]), (5, [5.0])]:
        with pytest.raises(ValueError):
            fastrace.forward_and_slopes_base(*args, bid, prm)
