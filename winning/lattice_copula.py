import numpy as np
from statistics import NormalDist
import math
from winning.lattice import cdf_to_pdf, pdf_to_cdf, state_prices_from_densities, five_prices_from_five_densities
from winning.lattice_plot import densitiesPlot

try:
    from scipy.integrate import quad_vec
    using_scipy = True
except ImportError:
    print('pip install --upgrade scipy')
    using_scipy = False

if using_scipy:

    std_normal_dist = NormalDist(mu=0, sigma=1.0)

    def safe_cond_cdf(c,rho,z):
        if c<1e-6:
            return c
        elif c>1-1e-6:
            return c
        else:
            return std_normal_dist.cdf(   ( std_normal_dist.inv_cdf( c ) - rho*z )/math.sqrt(1-rho))

    def gaussian_copula_conditional_cdf( cdf, rho:float, z:float ):
        """
            Condition a densities' cdf on common factor
        """
        if abs(rho)<1e-6:
            return [ c for c in cdf ]
        else:
            return [ safe_cond_cdf(c=c,rho=rho,z=z) for c in cdf ]


    def gaussian_copula_functional(densities, rho, f):
        """
        :param densities:  [ [float] ]   performance densities
        :param f:          Some linear functional acting on a density collection
        :return:
        """
        cdfs = [ pdf_to_cdf( pdf ) for pdf in densities ]

        def conditional_functional(p):
            z = std_normal_dist.inv_cdf(p=p)
            z_pdfs = [ cdf_to_pdf( gaussian_copula_conditional_cdf(cdf=cdf,rho=rho, z=z)) for cdf in cdfs ]
            return f(z_pdfs)

        res, err = quad_vec(f=conditional_functional, a=1e-6,b=1-1e-6, epsabs=1e-8, epsrel=1e-6)
        return res

    def gaussian_copula_win(densities, rho:float):
        """
            Rank 1 probabilities
            :param rho is gaussian correlation
        """

        def conditional_win(ds):
            return np.asarray( state_prices_from_densities(ds) )

        return gaussian_copula_functional(densities=densities, f=conditional_win, rho=rho)


    def gaussian_copula_five(densities, rho:float):
        """
        :param densities:
        :param rho: gaussian correlation
        :returns:  [ [first probs] ... [fifth probs] ]
        """
        assert len(densities)==5

        def conditional_five(ds):
            fv = five_prices_from_five_densities(ds)
            return np.asarray( [ p for pos in fv for p in pos ])

        cfv = gaussian_copula_functional(densities=densities, f=conditional_five, rho=rho)
        return [ cfv[i:i+5] for i in range(0,25,5)]





