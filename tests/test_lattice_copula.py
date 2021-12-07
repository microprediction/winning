from winning.lattice_copula import gaussian_copula_win
from winning.lattice import skew_normal_density
from winning.lattice_plot import densitiesPlot


def test_ensure_scipy():
    from winning.scipyinclusion import using_scipy
    assert using_scipy


def test_two_skew():
    density1 = skew_normal_density(L=50, unit=0.1, scale=1.0, loc=1.0, a=1.0)
    density2 = skew_normal_density(L=50, unit=0.1, scale=1.0, loc=0, a=1.0)
    densities = [ density1, density2 ]
    state_prices = gaussian_copula_win( densities=densities, rho=0.85 )
    return densities, state_prices



if __name__=='__main__':
    densities, state_prices = test_two_skew()
    densitiesPlot(densities=densities,unit=0.1)
    print(state_prices)

