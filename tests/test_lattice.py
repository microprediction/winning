from winning.lattice import skew_normal_density, winner_of_many, implicit_state_prices, \
    sample_winner_of_many, densities_from_offsets, state_prices_from_offsets
from winning.lattice_calibration import solve_for_implied_offsets

unit = 0.1
L = 150

def test_minimumPdf():
    skew1 = skew_normal_density(L=L, unit=unit, a=1.5)
    skew2 = skew_normal_density(L=L, unit=unit, a=1.5, loc=-0.5)
    skew3 = skew_normal_density(L=L, unit=unit, a=1.5, loc=-1.0)
    best, multiplicity = winner_of_many([skew1, skew2, skew3])


def test_implicit_payoffs():
    skew1 = skew_normal_density(L=L, unit=unit, a=1.5)
    skew2 = skew_normal_density(L=L, unit=unit, a=1.5, loc=-0.5)
    skew3 = skew_normal_density(L=L, unit=unit, a=1.5, loc=-1.0)
    densityAll, multiplicityAll = winner_of_many([skew1, skew2, skew3])
    payoffs = implicit_state_prices(density=skew1, densityAll=densityAll, multiplicityAll=multiplicityAll, cdf=None,
                                    cdfAll=None, offsets=None)
    # TODO: what's the test?


def test_monte_carlo():
    skew1 = skew_normal_density(L=25, unit=unit, a=1.5)
    densities = [skew1, skew1, skew1]
    densityAll, multiplicityAll = winner_of_many(densities)
    densityAllCheck = sample_winner_of_many(densities, nSamples=5000)
    assert all(abs(p1 - p2) < 3e-2 for p1, p2 in zip(densityAll, densityAllCheck))


def test_calibration():
    skew1 = skew_normal_density(L=L, unit=unit, a=1.5)
    prices = [0.2, 0.3, 0.5]
    implied_offsets = solve_for_implied_offsets(prices=prices, density=skew1, nIter=2)
    inferred_prices = state_prices_from_offsets(skew1, implied_offsets)
    print(str(inferred_prices))
    densities = densities_from_offsets(skew1, implied_offsets)
    densityAllAgain, multiplicityAll = winner_of_many(densities)
