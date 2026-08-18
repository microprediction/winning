"""Exact lattice win probabilities against closed forms."""

import math
from statistics import NormalDist

from winning import gaussian_win_probabilities


def test_two_runner_matches_closed_form():
    # P(A wins) = Phi((muA - muB) / sqrt(sA^2 + sB^2 + 2 beta^2))
    mu_a, mu_b = 1.0, 0.0
    s_a, s_b, beta = 0.5, 0.8, 1.0
    expected = NormalDist().cdf((mu_a - mu_b) / math.sqrt(s_a**2 + s_b**2 + 2 * beta**2))
    p = gaussian_win_probabilities([mu_a, mu_b], [s_a, s_b], beta=beta)
    assert abs(p[0] - expected) < 1e-3
    assert abs(sum(p) - 1.0) < 1e-9


def test_symmetric_field_is_uniform():
    p = gaussian_win_probabilities([0.0] * 5, [1.0] * 5, beta=1.0)
    assert all(abs(q - 0.2) < 1e-6 for q in p)


def test_monotone_in_mu():
    p = gaussian_win_probabilities([2.0, 1.0, 0.0], [1.0, 1.0, 1.0], beta=1.0)
    assert p[0] > p[1] > p[2]


def test_uncertainty_shrinks_edge():
    confident = gaussian_win_probabilities([1.0, 0.0], [0.1, 0.1], beta=1.0)
    uncertain = gaussian_win_probabilities([1.0, 0.0], [3.0, 3.0], beta=1.0)
    assert confident[0] > uncertain[0] > 0.5


def test_variance_validation():
    import pytest

    with pytest.raises(ValueError):
        # a zero-total-variance contestant in a field of 3+ has no density
        gaussian_win_probabilities([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], beta=0.0)
    with pytest.raises(ValueError):
        gaussian_win_probabilities([0.0, 0.0], [1.0, -1.0])


def test_heteroskedastic_field():
    # per-contestant noise with beta=0: erratic contestants win more from behind
    p = gaussian_win_probabilities([-0.5, -0.5, 0.5], [0.6, 1.6, 0.6], beta=0.0)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[1] > p[0]  # same mean, more variance -> more upset wins
    assert p[2] > p[1]


def test_two_runner_degenerate_and_fast_path():
    assert gaussian_win_probabilities([1.0, 0.0], [0.0, 0.0], beta=0.0) == [1.0, 0.0]
    assert gaussian_win_probabilities([0.5, 0.5], [0.0, 0.0], beta=0.0) == [0.5, 0.5]
