"""The pair inverse must not lose a tiny positive share to rounding of
its complement, and relabeling the pair must relabel the answer (#412).

For [1, 1e-16] the normalized first share rounds to exactly 1 and the
closed form inverted it: ndtri(1) = inf, certified converged, while the
permutation [1e-16, 1] returned finite abilities."""

import numpy as np
import pytest
from scipy.special import ndtri

from winning.factor.races import abilities_from_race, race_probabilities

TINY = [1e-12, 1e-16, 1e-20, 1e-50, 1e-300]


@pytest.mark.parametrize("t", TINY)
def test_both_label_orders_are_finite_and_mirror(t):
    D = np.array([0.5, 0.5])
    a, ia = abilities_from_race(np.array([t, 1.0]), D=D, return_info=True)
    b, ib = abilities_from_race(np.array([1.0, t]), D=D, return_info=True)
    assert np.isfinite(a).all() and np.isfinite(b).all(), (a, b)
    assert ia["converged"] and ib["converged"]
    assert np.allclose(b, a[::-1], rtol=1e-12, atol=0), (a, b)
    # the closed form: mu1 - mu0 = sd Phi^-1(p0), the small share exact
    want = 1.0 * ndtri(t / (1 + t))           # sd of the contrast is 1
    assert abs((a[1] - a[0]) - want) <= 1e-10 * abs(want)
    p = race_probabilities(b, D=D)
    assert abs(np.log(p[1]) - np.log(t)) < 1e-6, (p, t)


@pytest.mark.parametrize("pair", [([1e20, 1.0], [1.0, 1e20])])
def test_scaled_equivalents(pair):
    a = abilities_from_race(np.array(pair[0]))
    b = abilities_from_race(np.array(pair[1]))
    assert np.isfinite(a).all() and np.allclose(a, b[::-1], rtol=1e-12)


def test_target_floor_near_machine_precision():
    a, info = abilities_from_race(np.array([1.0, 0.0]), target_floor=1e-17,
                                  return_info=True)
    assert np.isfinite(a).all() and info["converged"]
    assert info["floored"].tolist() == [False, True]
