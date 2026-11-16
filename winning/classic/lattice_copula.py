import numpy as np
import math
from winning.classic.lattice import (cdf_to_pdf, pdf_to_cdf, state_prices_from_densities, five_prices_from_five_densities,
                                     as_unit_densities, _state_prices_from_densities)
from winning.classic.normaldist import normcdf, invnormcdf
from winning.classic.scipyinclusion import using_scipy

if using_scipy:
    from scipy.integrate import quad_vec

    def check_rho(rho: float) -> float:
        """ rho is the pairwise (equi)correlation of the contestants' Gaussian scores.

            The copula is the one-factor construction X_i = sqrt(rho) Z + sqrt(1-rho) eps_i,
            so Corr(X_i, X_j) = rho. It cannot represent a negative equicorrelation
            and rho = 1 is the degenerate common score, so 0 <= rho < 1 is required.
            rho itself used to be the loading, which made the realised correlation
            rho**2 and left rho and -rho indistinguishable (#910).
        """
        rho = float(rho)
        if not (math.isfinite(rho) and 0.0 <= rho < 1.0):
            raise ValueError(f"rho is the contestants' equicorrelation and must satisfy 0 <= rho < 1, got {rho!r}")
        return rho

    def safe_cond_cdf(c, rho, z):
        if c<1e-6:
            return c
        elif c>1-1e-6:
            return c
        else:
            return normcdf(   ( invnormcdf( c ) - math.sqrt(rho)*z )/math.sqrt(1-rho))

    def gaussian_copula_conditional_cdf( cdf, rho:float, z:float ):
        """
            Condition a densities' cdf on the common factor z
            :param rho  equicorrelation of the contestants, 0 <= rho < 1 (see check_rho)
        """
        rho = check_rho(rho)
        if rho<1e-12:
            return [ c for c in cdf ]
        else:
            return [ safe_cond_cdf(c=c,rho=rho,z=z) for c in cdf ]


    RHO_SPLIT = 0.999      # see gaussian_copula_functional (#942)

    def gaussian_copula_functional(densities, rho, f):
        """
        :param densities:  [ [float] ]   performance densities
        :param rho:        equicorrelation of the contestants' performances, 0 <= rho < 1
        :param f:          Some linear functional acting on a density collection
        :return:
        """
        rho = check_rho(rho)
        densities = as_unit_densities(densities)      # counts are their frequencies
        cdfs = [ pdf_to_cdf( pdf ) for pdf in densities ]

        def conditional_functional(p):
            z = invnormcdf(p)
            z_pdfs = [ cdf_to_pdf( gaussian_copula_conditional_cdf(cdf=cdf,rho=rho, z=z)) for cdf in cdfs ]
            return f(z_pdfs)

        # As rho -> 1 each conditional cdf is a step at
        # z = invnormcdf(c) / sqrt(rho), i.e. p = normcdf(invnormcdf(c) /
        # sqrt(rho)), for every marginal cdf value c the conditioning moves.
        # One adaptive pass over (0, 1) could step over those regions
        # entirely, and two distinct histograms came back an exact 50/50
        # race at rho = 0.999999 where the answer is [0.51, 0.49] (#942).
        # Splitting the integral at the breakpoints keeps every step on a
        # subinterval edge. Below rho = 0.999 the unsplit pass is accurate
        # (worst 2.5e-7 over random discrete marginals; 1e-5 at 0.9999,
        # 7.5e-3 at 0.999999) and a lattice density's thousand breakpoints
        # cost 25-35x, so the split is kept for the near-degenerate regime.
        a, b = 1e-12, 1 - 1e-12
        breaks = []
        if rho >= RHO_SPLIT:
            sr = math.sqrt(rho)
            for cdf in cdfs:
                for c in cdf:
                    if 1e-6 <= c <= 1 - 1e-6:
                        p = normcdf(invnormcdf(c) / sr)
                        if a < p < b:
                            breaks.append(p)
        breaks = sorted(set(breaks))
        # split: each step region gets its own error budget, so the global
        # tolerance is tightened with it -- at the unsplit tolerance the
        # split pass was LESS accurate at rho = 0.999 (3.3e-5 vs 2.5e-7)
        tol = (1e-10, 1e-8) if breaks else (1e-6, 1e-3)
        I1, err1 = quad_vec(f=conditional_functional, a=a, b=b, epsabs=tol[0], epsrel=tol[1],
                            points=breaks or None)
        return I1

    def gaussian_copula_win(densities, rho:float):
        """
            Rank 1 probabilities
            :param rho  pairwise Gaussian correlation of the performances, 0 <= rho < 1
        """

        def conditional_win(ds):
            return np.asarray( _state_prices_from_densities(ds) )

        return gaussian_copula_functional(densities=densities, f=conditional_win, rho=rho)


    def gaussian_copula_five(densities, rho:float):
        """
        :param densities:
        :param rho: pairwise Gaussian correlation of the performances, 0 <= rho < 1
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