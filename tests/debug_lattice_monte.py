from winning.lattice import skew_normal_density, winner_of_many, implicit_state_prices, \
    sample_winner_of_many, densities_from_offsets, state_prices_from_offsets
from winning.lattice_calibration import solve_for_implied_offsets

unit = 0.1
L = 150


def debug_monte_carlo():
    skew1 = skew_normal_density(L=25, unit=unit, a=1.5)
    densities = [skew1, skew1, skew1]
    densityAll, multiplicityAll = winner_of_many(densities)
    densityAllCheck = sample_winner_of_many(densities, nSamples=5000)
    assert all(abs(p1 - p2) < 3e-2 for p1, p2 in zip(densityAll, densityAllCheck))

if __name__=='__main__':
    debug_monte_carlo()