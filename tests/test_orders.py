"""The order convention: best-first permutation, never a rank array."""
import numpy as np
import pytest

from winning.ratings import (order_from_performance, order_from_positions,
                             order_from_times, positions_from_order,
                             update_ranking_exact)


def test_worked_example_from_the_report():
    perf = [0.2, 0.9, -0.4, 0.5]
    order = order_from_performance(perf)
    assert order.tolist() == [1, 3, 0, 2]
    assert positions_from_order(order).tolist() == [2, 0, 3, 1]
    assert order_from_positions([2, 0, 3, 1]).tolist() == [1, 3, 0, 2]
    assert order_from_positions([3, 1, 4, 2]).tolist() == [1, 3, 0, 2]   # 1-based
    assert order_from_times([10.2, 9.8, 11.0]).tolist() == [1, 0, 2]


def test_round_trip_and_pairwise_coincidence():
    rng = np.random.default_rng(3)
    for K in (2, 3, 7, 20):
        order = rng.permutation(K)
        assert order_from_positions(positions_from_order(order)).tolist() == order.tolist()
    # at K = 2 the two conventions are the same array
    assert order_from_positions([1, 0]).tolist() == [1, 0]
    assert positions_from_order(np.array([1, 0])).tolist() == [1, 0]


def test_ties_and_bad_inputs_raise():
    with pytest.raises(ValueError):
        order_from_positions([1, 1, 2])
    with pytest.raises(ValueError):
        order_from_performance([0.0, np.nan])
    with pytest.raises(ValueError):
        positions_from_order([0, 2])          # partial order has no positions
    with pytest.raises(ValueError):
        positions_from_order([0.0, 1.0])


def _repeat(order, n=30, K=None):
    K = len(order) if K is None else K
    m, v = np.zeros(K), np.ones(K)
    for _ in range(n):
        m, v = update_ranking_exact(m, v, order, beta2=1.0)
    return m


def test_same_order_repeated_gives_means_decreasing_along_it():
    order = [0, 1, 2, 3, 4, 5]
    m = _repeat(order)
    assert np.all(np.diff(m[order]) < 0)
    assert np.allclose(m, -m[::-1], atol=1e-6)                 # symmetric field


def test_rank_array_passed_as_order_is_a_different_observation():
    order = np.array([1, 3, 0, 2])                              # not self-inverse
    positions = positions_from_order(order)
    m_right = _repeat(order)
    m_wrong = _repeat(positions)
    assert np.all(np.diff(m_right[order]) < 0)
    assert not np.allclose(m_right, m_wrong, atol=1e-3)
