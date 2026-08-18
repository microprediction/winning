import numpy as np

from winning.thurstone import AbilityCalibrator, Density, UniformLattice


def test_basic_flow():
    lat = UniformLattice(100, 0.1)
    base = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
    divs = [3.0, 4.0, 8.0, 10.0]
    cal = AbilityCalibrator(base, n_iter=2)
    ability = cal.solve_from_dividends(divs)
    prices = cal.state_prices_from_ability([a * lat.unit for a in ability])
    assert np.isclose(sum(prices), 1.0, atol=1e-6)
