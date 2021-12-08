from winning.lattice_copula import gaussian_copula_five
from winning.lattice import skew_normal_density
from winning.lattice_plot import densitiesPlot
from pprint import pprint



def test_ensure_scipy():
    from winning.scipyinclusion import using_scipy
    from scipy.integrate import quad_vec
    assert using_scipy


def test_five_skew():
    mus = [-0.5, -0.25, 0, 1, 1.5]
    scales = [1.0, 1.5, 1.2, 1.3, 2.0]
    densities = [skew_normal_density(L=100, unit=0.1, scale=scale, loc=mu, a=1.0) for mu, scale in zip(mus, scales)]
    rank_probs = gaussian_copula_five(densities, rho=0.01)
    return densities, rank_probs



if __name__=='__main__':
    densities, rank_probs = test_five_skew()
    legend = ['Asset ' + str(i) for i in range(1, 6)]
    densitiesPlot(densities=densities, unit=0.1, legend=legend)
    pprint(rank_probs)

