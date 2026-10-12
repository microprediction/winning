"""Regression tests for the factor-core round-3 issue batch."""
import warnings

import numpy as np
import pytest

from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes, win_probabilities_factor)


# #374: the pair closed form is exact only for a Gaussian factor law; a
# centred two-point rule has the same two moments and a different map.
@pytest.mark.parametrize("F, D, truth", [
    ([[-1.0], [1.0]], 0.05, 0.4),     # centred, mean zero, variance one
    ([[0.0], [2.0]], 0.2, 0.3),       # translated
])
def test_pair_inverse_is_certified_by_the_forward(F, D, truth):
    V = np.array([[0.0], [1.0]]); D = np.full(2, D)
    F = np.array(F); W = np.array([0.5, 0.5])
    t = win_probabilities_factor(np.array([-truth, truth]), V, D, F, W,
                                 points=1025)
    mu, info = abilities_from_probabilities_factor(t, V, D, F, W,
                                                   points=1025,
                                                   return_info=True)
    r = win_probabilities_factor(mu, V, D, F, W, points=1025)
    actual = float(np.abs(np.log(r / t)).max())
    assert info["converged"] and actual < 1e-6
    assert abs(info["residual"] - actual) < 1e-9
    assert np.allclose(mu, [-truth, truth], atol=1e-6)


def test_pair_inverse_on_a_hermite_rule_round_trips():
    V = np.array([[0.0], [1.0]]); D = np.full(2, 0.05)
    F, W = hermite_nodes(1)
    t = win_probabilities_factor(np.array([-0.3, 0.3]), V, D, F, W)
    mu, info = abilities_from_probabilities_factor(t, V, D, F, W,
                                                   return_info=True)
    assert info["converged"]
    assert np.allclose(mu, [-0.3, 0.3], atol=1e-5)
