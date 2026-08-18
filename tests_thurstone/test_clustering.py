from numpy.testing import assert_allclose

from winning.thurstone import AbilityCalibrator


def test_right_hanger_gets_zero_share(base):
    cal = AbilityCalibrator(base)
    # very slow entrant on the right
    prices = cal.state_prices_from_ability([0.0, 0.0, +80.0])
    assert_allclose(sum(prices), 1.0, atol=1e-12)
    assert prices[-1] == 0.0
    assert abs(prices[0] - prices[1]) < 1e-12  # symmetry of the two at 0


def test_left_hanger_sweeps(base):
    cal = AbilityCalibrator(base)
    prices = cal.state_prices_from_ability([-80.0, 0.0, 0.0])
    assert_allclose(sum(prices), 1.0, atol=1e-12)
    assert prices[0] == 1.0 and prices[1] == prices[2] == 0.0


def test_both_hangers_symmetric_allocation(base):
    cal = AbilityCalibrator(base)
    a = [-60.0, 0.0, +60.0]
    p = cal.state_prices_from_ability(a)
    # fast left wins ~1; center and very slow get ~0
    assert_allclose(sum(p), 1.0, atol=1e-12)
    assert p[0] > 0.999 and p[1] < 1e-6 and p[2] < 1e-12


def test_translation_invariance(base):
    cal = AbilityCalibrator(base)
    a = [-6.0, -3.0, 0.0, +2.0]
    p1 = cal.state_prices_from_ability(a)
    p2 = cal.state_prices_from_ability([ai + 7.0 for ai in a])
    # adding a constant to all abilities doesn’t change prices
    assert_allclose(p1, p2, atol=1e-12)


def test_max_depth_terminates(base):
    cal = AbilityCalibrator(base)
    # absurd offsets; should still return quickly and yield a valid distribution
    p = cal.state_prices_from_ability([-1e6, 0.0, +1e6])
    assert_allclose(sum(p), 1.0, atol=1e-12)
    assert p[0] == 1.0 and p[-1] == 0.0
