from winning.lattice import skew_normal_density
from winning.lattice_plot import densitiesPlot
from pprint import pprint
from winning.scipyinclusion import using_scipy
if using_scipy:
    from scipy.integrate import quad_vec
    from winning.lattice_copula import gaussian_copula_five


    def test_five_skew():
        _,_ = do_five_skew()


    def do_five_skew():
        mus = [-0.5, -0.25, 0, 1, 1.5]
        scales = [1.0, 1.5, 1.2, 1.3, 2.0]
        densities = [skew_normal_density(L=100, unit=0.1, scale=scale, loc=mu, a=1.0) for mu, scale in zip(mus, scales)]
        rank_probs = gaussian_copula_five(densities, rho=0.01)
        return densities, rank_probs



if __name__=='__main__':
    if using_scipy:
        import time
        st = time.time()
        densities, rank_probs = do_five_skew()
        legend = ['Asset ' + str(i) for i in range(1, 6)]
        print({'elapsed':time.time()-st})
        densitiesPlot(densities=densities, unit=0.1, legend=legend)
        pprint(rank_probs)

