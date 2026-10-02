"""Harville top-k at the simplex boundary (#384)."""
import numpy as np
import pytest

from winning.factor.races import (plackett_luce_topk_probabilities,
                                  softmax_probabilities)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_k_at_least_n_is_certain_membership(n):
    p = softmax_probabilities(np.r_[1000.0, np.zeros(n - 1)])
    for k in range(n, 4):
        np.testing.assert_array_equal(plackett_luce_topk_probabilities(p, k),
                                      np.ones(n))


def test_softmax_underflow_does_not_lose_membership():
    p = softmax_probabilities(np.array([1000.0, 0.0, 0.0]))
    assert p[0] == 0.0
    for k in (1, 2, 3):
        assert plackett_luce_topk_probabilities(p, k).sum() == pytest.approx(k)


@pytest.mark.parametrize("k", [2, 3])
def test_exact_zero_is_the_limit_of_vanishing_weight(k):
    tiny = plackett_luce_topk_probabilities(np.array([1e-300, 1, 1, 1, 2.0]), k)
    zero = plackett_luce_topk_probabilities(np.array([0.0, 1, 1, 1, 2.0]), k)
    np.testing.assert_allclose(zero, tiny, atol=1e-12)
    assert zero.sum() == pytest.approx(k)


@pytest.mark.parametrize("k", [2, 3])
def test_a_lone_positive_runner_leaves_the_rest_to_chance(k):
    q = plackett_luce_topk_probabilities(np.array([0.0, 0.0, 1.0, 0.0, 0.0]), k)
    np.testing.assert_allclose(q, [(k - 1) / 4] * 2 + [1.0] + [(k - 1) / 4] * 2)
    assert q.sum() == pytest.approx(k)


def test_k_out_of_range_is_refused():
    with pytest.raises(ValueError):
        plackett_luce_topk_probabilities(np.ones(6) / 6, 4)
    with pytest.raises(ValueError):
        plackett_luce_topk_probabilities(np.ones(6) / 6, 0)
