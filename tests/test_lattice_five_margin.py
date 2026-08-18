from winning.lattice import skew_normal_density
from winning.lattice_plot import densitiesPlot
from winning.scipyinclusion import using_scipy

if using_scipy:
    from scipy.integrate import quad_vec
    assert using_scipy
    from winning.lattice_copula import gaussian_copula_margin_0


    def test_five_skew():
        do_five_skew()


    def do_five_skew():
        mus = [-0.5, -0.25, 0, 1, 1.5]
        scales = [1.0, 1.5, 1.2, 1.3, 2.0]
        densities = [skew_normal_density(L=500, unit=0.01, scale=scale, loc=mu, a=1.0) for mu, scale in zip(mus, scales)]
        margin_0 = gaussian_copula_margin_0(densities, rho=0.9)
        return densities[0], margin_0


if __name__ == '__main__':
    if using_scipy:
        density1, density2 = do_five_skew()
        legend = ['margin', 'reconstructed']
        densitiesPlot(densities=[density1, density2], unit=0.1, legend=legend)
        print(sum(density2))
