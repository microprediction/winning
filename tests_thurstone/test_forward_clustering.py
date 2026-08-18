import numpy as np
from numpy.testing import assert_allclose

from winning.thurstone import AbilityCalibrator

ATOL = 1e-6


def test_extreme_best_and_worst(base):
    cal = AbilityCalibrator(base)
    # One very strong, one middle, one very weak
    ability = [-50.0, 0.0, +50.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    # The very strong runner should essentially win
    assert p[0] > 0.999
    # The very weak runner should be negligible
    assert p[2] < 1e-6
    # Middle runner is "almost irrelevant"
    assert p[1] < 1e-3


def test_right_hangers_share_zero(base):
    cal = AbilityCalibrator(base)
    # Two reasonable, one absurdly slow
    ability = [-1.0, +1.0, +100.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    # slowest gets no realistic chance
    assert p[2] == 0.0
    # the first two share the pot
    assert p[0] > 0.0 and p[1] > 0.0
    assert_allclose(p[0] + p[1], 1.0, atol=ATOL)


def test_both_sides_hanging_clusters(base):
    cal = AbilityCalibrator(base)
    # Fast cluster (left), mid cluster, slow cluster (right)
    ability = [-40.0, -39.0, 0.0, 1.0, 40.0, 41.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    # Left cluster dominates, right cluster almost dead
    left_share = p[0] + p[1]
    mid_share = p[2] + p[3]
    right_share = p[4] + p[5]
    assert left_share > 0.99
    assert right_share < 1e-6
    assert mid_share < 1e-3
    # Within the left cluster, better runner should have higher prob
    assert p[0] > p[1]


def test_all_extremely_bad_walkover(base):
    cal = AbilityCalibrator(base)
    # All hopeless but not equal: the least-bad should take (walkover)
    ability = [100.0, 120.0, 130.0, 200.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    assert p[0] > 0.99
    for i in range(1, 4):
        assert p[i] < 1e-6


def test_all_extremely_good_walkover(base):
    cal = AbilityCalibrator(base)
    # All great but not equal: the best should take (walkover)
    ability = [-200.0, -150.0, -120.0, -100.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    assert p[0] > 0.99
    for i in range(1, 4):
        assert p[i] < 1e-6


def test_group_shift_does_not_break_others(base):
    cal = AbilityCalibrator(base)
    # baseline group: 4 runners in a reasonable range
    base_ability = [-2.0, -1.0, 0.0, +1.0]
    p0 = cal.state_prices_from_ability(base_ability)
    # Now add a cluster of very slow runners
    ability = base_ability + [20.0, 21.0, 22.0, 23.0]
    p = cal.state_prices_from_ability(ability)
    assert_allclose(sum(p), 1.0, atol=ATOL)
    # The slow cluster should get almost no share
    slow_share = sum(p[4:])
    assert slow_share < 1e-4
    # The original four should still have nearly the same relative probabilities
    scaled = np.array(p[:4]) / sum(p[:4])
    assert_allclose(scaled, p0, atol=1e-2)
