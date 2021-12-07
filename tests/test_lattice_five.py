from winning.lattice_copula import gaussian_copula_five
from winning.lattice import skew_normal_density
from winning.lattice_plot import densitiesPlot
from pprint import pprint



def test_ensure_scipy():
    from winning.scipyinclusion import using_scipy
    from scipy.integrate import quad_vec
    assert using_scipy


def test_five_skew():
    densities = [ skew_normal_density(L=100, unit=0.1, scale=1.0, loc=i-2, a=1.0) for i in range(5) ]
    rank_probs = gaussian_copula_five(densities, rho=0.2)
    return densities, rank_probs



if __name__=='__main__':
    densities, rank_probs = test_five_skew()
    densitiesPlot(densities=densities,unit=0.1)
    pprint(rank_probs)

