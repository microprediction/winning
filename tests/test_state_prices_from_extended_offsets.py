from winning.lattice import state_prices_from_offsets, state_prices_from_extended_offsets
from winning.std_calibration import centered_std_density
import numpy as np


def prices_almost_same(p1, p2, tol=1e-6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def test_race_with_one_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    finite_prices = state_prices_from_offsets(offsets=finite_offsets, density=density)
    expected_prices = list(finite_prices) + [0.]
    offsets = finite_offsets + [float('inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_race_with_two_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    finite_prices = state_prices_from_offsets(offsets=finite_offsets, density=density)
    expected_prices = list(finite_prices) + [0., 0.]
    offsets = finite_offsets + [float('inf'), float('inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_race_with_one_neg_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    expected_prices = [0 for _ in finite_offsets] + [1.0]
    offsets = finite_offsets + [float('-inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_race_with_two_neg_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    expected_prices = [0 for _ in finite_offsets] + [0.5, 0.5]
    offsets = finite_offsets + [float('-inf'), float('-inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_race_with_two_neg_inf_and_one_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    expected_prices = [0 for _ in finite_offsets] + [0.5, 0.5] + [0.]
    offsets = finite_offsets + [float('-inf'), float('-inf'), float('inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_race_with_two_neg_inf_and_three_inf():
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_offsets = [-5, -3.5, 2.0, 1.0]
    expected_prices = [0 for _ in finite_offsets] + [0.5, 0.5] + [0., 0., 0.]
    offsets = finite_offsets + [float('-inf'), float('-inf'), float('inf'), float('inf'), float('inf')]
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def assert_heterogenous_race(finite_offsets, n_huge=1, n_inf=1, n_neg_inf=3):
    HUGE = 500
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    finite_prices = state_prices_from_offsets(density=density, offsets=finite_offsets)
    offsets = finite_offsets + [HUGE] * n_huge + [float('inf')] * n_inf + [float('-inf')] * n_neg_inf
    if n_neg_inf == 0:
        expected_prices = finite_prices + [0.] * n_huge + [0.] * n_inf
    else:
        expected_prices = [0] * len(finite_offsets) + [0] * n_huge + [0] * n_inf + [1 / n_neg_inf] * n_neg_inf
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets)
    assert prices_almost_same(p1=prices, p2=expected_prices)


def test_heterogenous_race1():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=0, n_inf=0)


def test_heterogenous_race2():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=0, n_inf=0)


def test_heterogenous_race3():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=1, n_inf=0)


def test_heterogenous_race4():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=0, n_inf=1)


def test_heterogenous_race3():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=1, n_inf=0)


def test_heterogenous_race4():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=0, n_inf=1)


def test_heterogenous_race5():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=1, n_inf=1)


def test_heterogenous_race6():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=1, n_inf=1)


def test_heterogenous_race7():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=2, n_inf=0)


def test_heterogenous_race8():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=1, n_inf=0)


def test_heterogenous_race9():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=2, n_inf=0)


def test_heterogenous_race10():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=1, n_inf=1)


def test_heterogenous_race11():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=2, n_inf=0)


def test_heterogenous_race12():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=1, n_inf=1)


def test_heterogenous_race13():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=0, n_neg_inf=2, n_inf=1)


def test_heterogenous_race14():
    assert_heterogenous_race(finite_offsets=[0.1, -0.5, 4, 4.5], n_huge=1, n_neg_inf=2, n_inf=1)


def assert_split_race(offsets_1, offsets_2, n_huge=1, n_inf=1, n_neg_inf=3):
    from winning.lattice import int_centered
    HUGE = 500
    density = centered_std_density(L=21, unit=1.0, scale=1.0)
    offsets = offsets_1 + offsets_2 + [HUGE] * n_huge + [float('inf')] * n_inf + [float('-inf')] * n_neg_inf
    prices = state_prices_from_extended_offsets(density=density, offsets=offsets, max_depth=1)

    # Use a different (poor) approximation if the second group is small
    best_in_2 = min(offsets_2)
    group_1 = offsets_1+[best_in_2]
    # Try to approximate if we don't hit lattice issues
    try:
        assert len(offsets_2)<10, 'dont try this'
        finite_prices_12 = state_prices_from_offsets(density=density, offsets=group_1)
        share_2 = finite_prices_12[-1]
        share_1 = 1 - share_2
        finite_prices_1 = state_prices_from_offsets(density=density, offsets=int_centered(offsets_1))
        finite_prices_2 = state_prices_from_offsets(density=density, offsets=int_centered(offsets_2))
        got_em=True
    except:
        got_em=False

    if got_em:
        prices_1 = [share_1 * p for p in finite_prices_1]
        prices_2 = [share_2 * p for p in finite_prices_2]
        if n_neg_inf == 0:
            expected_prices = prices_1 + prices_2 + [0.] * n_huge + [0.] * n_inf
        else:
            expected_prices = [0] * len(offsets_1) +  [0]*len(offsets_2)+ [0] * n_huge + [0] * n_inf + [1 / n_neg_inf] * n_neg_inf
        assert prices_almost_same(p1=prices, p2=expected_prices)


def test_split_race_0():
    assert_split_race(offsets_1=[-30, -24, -27], offsets_2=[10,11,12], n_huge=1, n_neg_inf=1, n_inf=1)


def test_split_race_1():
    assert_split_race(offsets_1=[-30, -24, -27], offsets_2=[10,11,12], n_huge=1, n_neg_inf=2, n_inf=1)


def test_split_race_2():
    assert_split_race(offsets_1=[-30, -24, -27], offsets_2=[10,11,12], n_huge=1, n_neg_inf=1, n_inf=1)


def test_split_race_3():
    assert_split_race(offsets_1=[-30, -24, -27], offsets_2=[10,11,12], n_huge=1, n_neg_inf=5, n_inf=0)


def test_split_race_4():
    assert_split_race(offsets_1=[-30, -24, -27], offsets_2=[10,11,12], n_huge=1, n_neg_inf=0, n_inf=2)


def test_split_race_5():
    assert_split_race(offsets_1=[-30, -24, -28, 0,1,1], offsets_2=[10,11,12], n_huge=1, n_neg_inf=0, n_inf=2)

ABILITIES = [ 0.54415197,  0.33758834,  0.4457919 ,  0.09424374, -0.40353499,
        0.61386079,  0.15402363, -0.45340695,  0.44450927,  0.24130475,
       -0.54604444, -0.37272076, -0.48643902, -0.51578549,  0.15529696,
        0.27814857, -0.34811762,  0.51931145,  0.35561035,  0.00485529,
       -1.44361918,  0.39906961, -0.01809818]


def test_bigger_race_0():
    offsets_1 = ABILITIES
    offsets_2 = [a+10.0 for a in ABILITIES]
    assert_split_race(offsets_1=ABILITIES, offsets_2=offsets_2, n_neg_inf=0)


def test_bigger_race_1():
    offsets_1 = ABILITIES
    offsets_2 = [a+100.0 for a in ABILITIES]
    assert_split_race(offsets_1=offsets_1, offsets_2=offsets_2, n_neg_inf=0)


def test_huge_race_2():
    offsets_1 = ABILITIES
    offsets_2 = [ a+20 for a in np.random.randn(1000) ]
    assert_split_race(offsets_1=offsets_1, offsets_2=offsets_2, n_neg_inf=0,n_inf=0, n_huge=0)




if __name__ == '__main__':
    test_huge_race_2()
