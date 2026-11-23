"""A one-runner field wins with certainty under every registered method (#814).

Minimax tilting indexed mu with np.array([]) -- a float array -- and
raised IndexError on a single runner, where the other eleven methods
returned [1.0].
"""
import numpy as np
import pytest
from scipy.stats import norm

import winning.methods  # noqa: F401  (registers the methods)
from winning.methods.registry import METHODS


@pytest.mark.parametrize("name", sorted(METHODS))
def test_a_single_runner_wins_with_certainty(name):
    out = METHODS[name](np.array([0.0]), np.empty((1, 0)), np.array([1.0]),
                        budget=8, seed=7)
    p = np.asarray(out[0])
    assert p.shape == (1,) and p.dtype == np.float64
    assert p[0] == 1.0


def test_tilting_single_runner_info_matches_its_usual_info():
    _, info = METHODS["tilting"](np.array([0.3]), np.array([[0.5]]),
                                 np.array([2.0]), budget=8)
    assert info == {"draws": 8}


def test_tilting_single_runner_still_validates_its_budget():
    with pytest.raises(ValueError, match="budget"):
        METHODS["tilting"](np.array([0.0]), np.empty((1, 0)),
                           np.array([1.0]), budget=0)


def test_tilting_two_runners_unchanged():
    # nearby ordinary field: two independent unit runners, so
    # P(runner 0) = Phi((mu0 - mu1) / sqrt(2)), as mendell_elston gives
    mu = np.array([0.0, 0.4])
    p, info = METHODS["tilting"](mu, np.empty((2, 0)), np.ones(2),
                                 budget=4000, seed=7)
    assert info == {"draws": 4000}
    assert abs(p[0] - norm.cdf(-0.4 / np.sqrt(2))) < 1e-12
    p2, _ = METHODS["tilting"](mu, np.empty((2, 0)), np.ones(2),
                               budget=4000, seed=7)
    assert np.array_equal(p, p2)
