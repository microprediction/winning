import numpy as np
import math
from winning.lattice import cdf_to_pdf, pdf_to_cdf, state_prices_from_densities, five_prices_from_five_densities
from winning.normaldist import normcdf, invnormcdf
from winning.scipyinclusion import using_scipy

if using_scipy:
    from scipy.integrate import quad_vec

    def safe_cond_cdf(c,rho,z):
        if c<1e-6:
            return c
        elif c>1-1e-6:
            return c
        else:
            return normcdf(   ( invnormcdf( c ) - rho*z )/math.sqrt(1-rho*rho))

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
            z = invnormcdf(p)
            z_pdfs = [ cdf_to_pdf( gaussian_copula_conditional_cdf(cdf=cdf,rho=rho, z=z)) for cdf in cdfs ]
            return f(z_pdfs)

        I1, err1 = quad_vec(f=conditional_functional, a=1e-12,b=1-1e-12, epsabs=1e-6, epsrel=1e-3)
        return I1

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

    def gaussian_copula_margin_0(densities, rho:float):
        """
            Returns margin of the first, as a check
        """

        def conditional_margin_0(ds):
            return np.asarray( ds[0] )

        return gaussian_copula_functional(densities=densities, f=conditional_margin_0, rho=rho)

    try:
        import pandas as pd

        def five_to_df(rank_probs, names:[str]=None):
            """
                Convert the output of gaussian_copula_five into M2 format
            """
            if names is None:
                names = ['Asset '+str(i) for i in  range(1,6)]
            assert len(names)==5
            index = ['Rank ' + str(i) for i in range(1, 6)]
            df = pd.DataFrame(columns=names, data=rank_probs, index=index).transpose()
            return df

    except ImportError:
        pass