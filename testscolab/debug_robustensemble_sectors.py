from precise.skaters.managers.hrpmanagers import hrp_weak_weak_pm_t0_d0_r025_n50_s5_long_manager
from precise.skaters.portfoliostatic.ppoportfactory import ppo_vol_port

from testscolab.debug_robustensemble import generate_true_corr_matrix
from winning.scipyinclusion import using_scipy

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


if using_precise and using_scipy and using_sklearn:
    from sklearn.covariance import LedoitWolf, MinCovDet
    from winning.std_calibration import std_ability_implied_state_prices, std_state_price_implied_ability
    from scipy.stats import norm
    from typing import List
    import numpy as np
    from precise.skaters.portfoliostatic.diagport import diag_long_port
    from precise.skaters.covarianceutil.covfunctions import cov_to_corrcoef, multiply_off_diag, nearest_pos_def
    from precise.skaters.portfoliostatic.unitport import unit_port, unit_port_p050
    from precise.skaters.portfoliostatic.weakport import weak_long_port
    from precise.skaters.portfoliostatic.hrpport import hrp_diag_diag_s5_long_port, hrp_diag_weak_s5_long_port
    from scipy.stats import wishart
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    import numpy as np
    from collections import defaultdict

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*covariance matrix associated to your dataset is not full rank.*")

    from debug_robustensemble import ability_port, simulate_data, estimate_covariance, update_leaderboard, compute_portfolio_variance

    def generate_sector_cov_matrix(n_assets_per_sector: int, n_sectors: int, intra_corr: float,
                                   inter_corr: float) -> np.ndarray:
        """
        Generate a covariance matrix with block structure, where assets within the same sector
        have higher correlation (intra_corr) than assets across sectors (inter_corr).

        :param n_assets_per_sector: Number of assets per sector
        :param n_sectors: Number of sectors
        :param intra_corr: Correlation between assets within the same sector
        :param inter_corr: Correlation between assets across different sectors
        :return: Covariance matrix (n_assets x n_assets)
        """
        n_assets = n_assets_per_sector * n_sectors

        # Initialize the correlation matrix
        corr_matrix = generate_true_corr_matrix(n_assets)

        # Set intra-sector correlations
        for sector in range(n_sectors):
            start = sector * n_assets_per_sector
            end = start + n_assets_per_sector
            intra_sector_corr = np.random.rand()*intra_corr
            corr_matrix[start:end, start:end] = intra_sector_corr

        # Ensure diagonal elements are 1 (variance terms)
        np.fill_diagonal(corr_matrix, 1)

        # Random standard deviations for each asset (for the covariance matrix)
        std_devs = np.random.uniform(low=0.5, high=2.0, size=n_assets)

        # Convert correlation matrix to covariance matrix
        cov_matrix = corr_matrix * np.outer(std_devs, std_devs)

        return cov_matrix


    # Example: 5 sectors, each with 10 assets, intra-sector correlation 0.6, inter-sector correlation 0.2
    cov_matrix = generate_sector_cov_matrix(n_assets_per_sector=10, n_sectors=5, intra_corr=0.6, inter_corr=0.2)



def sector_comparison(ports: list,
                           n_sectors:int,
                           n_assets_per_sector:int,
                           n_data_samples: int,
                           intra_corr:float,
                           inter_corr:float,
                           n_iterations: int = 100, update_interval: int = 10, phi=0.25):
    """
    Compare different portfolio allocation methods over multiple iterations.

    :param ports: A dictionary where the key is the portfolio method name,
                       and the value is a function that computes the portfolio from cov
    :param n_dim: Number of dimensions for the covariance matrix (assets).
    :param n_data_samples: Number of samples used to estimate the covariance matrix.
    :param n_samples: Number of Monte Carlo samples for estimating ability-tilted portfolios.
    :param n_iterations: Number of iterations to perform.
    :param update_interval: How often to update the leaderboard (in iterations).
    :param phi: Multiplier applied to off-diagonal entries when computing ability portfolio.
    """

    # Dictionary to track variances for each portfolio method
    variances = defaultdict(list)

    # Run through multiple iterations
    for iteration in range(n_iterations):
        # Step 1: Generate true covariance matrix
        true_cov_matrix = generate_sector_cov_matrix(n_sectors=n_sectors,
                                                     n_assets_per_sector=n_assets_per_sector,
                                                     intra_corr=intra_corr,
                                                     inter_corr=inter_corr)
        from precise.skaters.covarianceutil.covfunctions import nearest_pos_def
        true_cov_matrix = nearest_pos_def(true_cov_matrix)
        print(f'True corr matrix ({n_sectors} sectors x {n_assets_per_sector} per sector)')
        print(cov_to_corrcoef(cov_matrix))
        print(f'True cov matrix ({n_sectors} sectors x {n_assets_per_sector} per sector)')
        print(cov_matrix)

        # Perform multiple rounds of data simulation and portfolio evaluation
        for _ in range(50):  # Repeat multiple times per iteration
            # Step 2: Simulate data based on the true covariance matrix
            data = simulate_data(true_cov_matrix, n_data_samples)

            # Step 3: Estimate correlation matrices from the simulated data
            estimates = estimate_covariance(data)
            cov_est = estimates['ledoitwolf_cov']
            cov_emp = nearest_pos_def(estimates['sample_cov'])

            # Step 4: Evaluate each portfolio method
            for  portfolio_func in ports:
                method_name = portfolio_func.__name__.replace('_long_port','').replace('_port','')

                # Step 4a: Compute portfolio weights using the method
                if 'ability' in method_name or 'hrp' in method_name:
                    w_portfolio = portfolio_func(cov=cov_emp)
                else:
                    w_portfolio = portfolio_func(cov=cov_est)

                w_sum = np.sum(w_portfolio)
                if abs(w_sum-1)>1e-3:
                    print(f'Problem with {method_name} as w_sum={w_sum}')
                    raise ValueError()

                # Step 5: Compute true portfolio variance using the true covariance matrix

                portfolio_variance = compute_portfolio_variance(w_portfolio, true_cov_matrix)

                # Store the variance for this method
                variances[method_name].append(portfolio_variance)

        # Periodically update and display the leaderboard
        if (iteration + 1) % update_interval == 0:
            update_leaderboard(variances, iteration + 1)


if __name__=='__main__':
    if using_precise and using_scipy and using_sklearn:

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

        ability_ports = [ equal_long_port, ability_1k_100_port, ability_1k_50_port, ability_1k_0_port]
        conventional_ports = [unit_port,
                              diag_long_port,
                              weak_long_port,
                              unit_port_p050,
                              hrp_diag_diag_s5_long_port]
        ports = ability_ports + conventional_ports
        n_sectors = 11
        n_assets_per_sector = 50
        n_data_samples = 20
        inter_corr = 0.2
        intra_corr = 0.9
        print('Starting...')
        sector_comparison(n_sectors=n_sectors, n_assets_per_sector=n_assets_per_sector,
                               inter_corr=inter_corr,
                               intra_corr=intra_corr,
                               ports=ports, n_data_samples=n_data_samples,
                          update_interval=1)