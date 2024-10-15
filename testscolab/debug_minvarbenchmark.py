from winning.scipyinclusion import using_scipy
import pandas as pd
pd.set_option('display.max_colwidth',200)

try:
    from precise.skaters.portfoliostatic.equalport import equal_long_port
    using_precise = True
except ImportError:
    using_precise = False

try:
    from sklearn.covariance import LedoitWolf, MinCovDet
    using_sklearn = True
except ImportError:
    using_sklearn = False

try:
    from randomcov import randomcovariancematrix
    using_randomcov = True
except:
    using_randomcov = False


if __name__=='__main__':
    print({'using_scipy':using_scipy,
           'using_precise':using_precise,
           'using_randomcov':using_randomcov,
           'using_sklearn':using_sklearn})

    if using_precise and using_scipy and using_sklearn and using_randomcov:

        from randomcov.benchmarkingutil.minvarleaderboard import min_var_leaderboard
        from precise.skaters.portfoliostatic.weakport import weak_long_port  # A long only portfolio
        from precise.skaters.portfoliostatic.unitport import unit_port, unit_port_p050  # The long/short min-var portfolio
        from precise.skaters.portfoliostatic.diagport import diag_long_port  # Ignore off-diagonal entries


        # Suppress specific warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*covariance matrix associated to your dataset is not full rank.*")


        from precise.skaters.covarianceutil.covfunctions import cov_to_corrcoef, multiply_off_diag

        import numpy as np
        from winning.std_calibration import std_ability_implied_state_prices, std_state_price_implied_ability

        from randomcov.corrgensutil.nearestcorr import nearest_corr


        def ability_port(cov, n_samples=10000, phi=1.0):
            w_diag = diag_long_port(cov=cov)
            corr = cov_to_corrcoef(cov)
            return ability_tilt(corr=corr, w=w_diag, n_samples=n_samples, phi=phi)


        def ability_tilt(corr: np.ndarray, w: np.ndarray, n_samples:int, phi:float) -> np.ndarray:
            """
                Tilt portfolio w using corr matrix
            """
            from randomcov.corrgensutil.isvalidcorr import is_valid_corr
            n_dim = np.shape(corr)[0]
            mu = std_state_price_implied_ability(w)
            fixed_corr = nearest_corr(corr)
            scaled_corr = multiply_off_diag(a=fixed_corr, phi=phi)  # sca
            if not is_valid_corr(scaled_corr):
                scaled_corr = nearest_corr(scaled_corr)

            correlated_samples = np.random.multivariate_normal(mu, scaled_corr, size=n_samples)
            min_indices = np.argmin(correlated_samples, axis=1)
            counts = np.bincount(min_indices, minlength=n_dim)
            return [c / sum(counts) for c in counts]


        def ability_1k_100_port(cov):
            return ability_port(cov=cov, n_samples=1000, phi=1.0)


        def ability_100k_100_port(cov):
            return ability_port(cov=cov, n_samples=100000, phi=1.0)


        def ability_1k_50_port(cov):
            return ability_port(cov=cov, n_samples=1000, phi=0.5)


        def ability_100k_50_port(cov):
            return ability_port(cov=cov, n_samples=100000, phi=0.5)


        def ability_1k_0_port(cov):
            return ability_port(cov=cov, n_samples=1000, phi=0.0)


        def ability_100k_0_port(cov):
            return ability_port(cov=cov, n_samples=100000, phi=0.0)


        ability_ports = [ equal_long_port, ability_1k_100_port, ability_1k_50_port, ability_1k_0_port, ability_100k_100_port, ability_100k_0_port]
        conventional_ports = [unit_port,
                              diag_long_port,
                              weak_long_port,
                              unit_port_p050]
        ports = ability_ports + conventional_ports

        min_var_leaderboard(n=20, ports=ports, n_data_samples=15, corr_method='lkj', var_method='lognormal',
                            n_inner_iter=10, update_interval=1)

