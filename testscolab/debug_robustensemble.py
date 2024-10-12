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
    from precise.skaters.covarianceutil.covfunctions import cov_to_corrcoef, multiply_off_diag
    from precise.skaters.portfoliostatic.unitport import unit_port, unit_port_p050
    from precise.skaters.portfoliostatic.weakport import weak_long_port
    from precise.skaters.portfoliostatic.hrpport import hrp_diag_diag_s5_long_port
    from scipy.stats import wishart
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*covariance matrix associated to your dataset is not full rank.*")


    def ability_port(cov, n_samples=10000, phi=0.25):
        w_diag = diag_long_port(cov=cov)
        corr = cov_to_corrcoef(cov)
        return ability_tilt(corr=corr, w=w_diag, n_samples=n_samples, phi=phi)


    def ability_tilt(corr: np.ndarray, w: np.ndarray, n_samples: int, phi=0.25) -> np.ndarray:
        """
            Tilt portfolio w using corr matrix
        """
        n_dim = np.shape(corr)[0]
        mu = std_state_price_implied_ability(w)
        scaled_corr = multiply_off_diag(a=corr, phi=phi)  # sca
        correlated_samples = np.random.multivariate_normal(mu, scaled_corr, size=n_samples)
        min_indices = np.argmin(correlated_samples, axis=1)
        counts = np.bincount(min_indices, minlength=n_dim)
        return [c / sum(counts) for c in counts]


    def generate_true_cov_matrix(n_dim: int) -> np.ndarray:
        """
        Generate a random covariance matrix with random variances.

        :param n_dim: Number of dimensions for the covariance matrix
        :return: Random covariance matrix (n_dim x n_dim)
        """
        corr_matrix = generate_true_corr_matrix(n_dim=n_dim)
        random_variances = np.random.uniform(low=0.1, high=10.0, size=n_dim)
        random_std_devs = np.sqrt(random_variances)  # Standard deviations from the random variances
        cov_matrix = corr_matrix * np.outer(random_std_devs, random_std_devs)

        return cov_matrix


    def generate_true_corr_matrix(n_dim: int) -> np.ndarray:
        """
        Generate a true correlation matrix using a random covariance matrix
        and normalize it to correlation.
        """
        # Generate a random covariance matrix via Wishart distribution
        cov_matrix = wishart.rvs(df=n_dim + 1, scale=np.eye(n_dim))

        # Convert covariance to correlation matrix
        stddevs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stddevs, stddevs)

        return corr_matrix


    def simulate_data(corr_matrix: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Simulate data from a multivariate normal distribution with the given correlation matrix.
        """
        n_dim = corr_matrix.shape[0]
        mean = np.zeros(n_dim)  # Mean is zero for all dimensions
        data = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)
        return data


    def estimate_correlation(data: np.ndarray) -> dict:
        """
        Estimate correlation matrix using different methods:
        1. Standard sample covariance
        2. Ledoit-Wolf shrinkage
        3. Minimum Covariance Determinant (robust)
        """
        # Sample covariance method
        sample_cov = np.cov(data, rowvar=False)
        sample_corr = np.corrcoef(data, rowvar=False)

        # Ledoit-Wolf shrinkage method
        lw = LedoitWolf()
        lw_cov = lw.fit(data).covariance_
        lw_corr = lw.covariance_ / np.outer(np.sqrt(np.diag(lw_cov)), np.sqrt(np.diag(lw_cov)))

        # Minimum Covariance Determinant (robust method)
        mcd = MinCovDet()
        mcd_cov = mcd.fit(data).covariance_
        mcd_corr = mcd.covariance_ / np.outer(np.sqrt(np.diag(mcd_cov)), np.sqrt(np.diag(mcd_cov)))

        return {
            'sample_corr': sample_corr,
            'ledoitwolf_corr': lw_corr,
            'mcd_corr': mcd_corr
        }

    def estimate_covariance(data: np.ndarray) -> dict:
        """
        Estimate correlation matrix using different methods:
        1. Standard sample covariance
        2. Ledoit-Wolf shrinkage
        3. Minimum Covariance Determinant (robust)
        """
        # Sample covariance method
        sample_cov = np.cov(data, rowvar=False)

        # Ledoit-Wolf shrinkage method
        lw = LedoitWolf()
        lw_cov = lw.fit(data).covariance_

        # Minimum Covariance Determinant (robust method)
        mcd = MinCovDet()
        mcd_cov = mcd.fit(data).covariance_

        return {
            'sample_cov': sample_cov,
            'ledoitwolf_cov': lw_cov,
            'mcd_cov': mcd_cov
        }


    def compute_portfolios(corr_matrix: np.ndarray, n_samples: int, port, phi=0.5):
        """
        Compute both unit_port and ability_tilt portfolios.
        """
        # Benchmark
        w_port = port(cov=corr_matrix)

        # Ability Tilt
        w_tilt = ability_port(corr_matrix, n_samples=n_samples, phi=phi)

        return w_port, w_tilt


    def compute_portfolio_variance(w: np.ndarray, true_corr_matrix: np.ndarray) -> float:
        """
        Compute the portfolio variance given the weights and the true correlation matrix.
        """
        return np.array(w).T @ true_corr_matrix @ np.array(w)


    def head_to_head_comparison(port, n_dim: int, n_data_samples, n_samples: int, n_iterations: int = 100, phi=0.25):
        """
        Compare different methods for portfolio allocation over multiple iterations.

           n_data_samples:  The number of samples we get to use when estimating cov
           n_samples:       The number of monte carlo samples used when computing the corr tilted ability portfolio
           phi:             The multiplier applied to the off-diagonal entries when computing the ability portfolio

        """
        # Initial equal-weight portfolio
        w0 = np.ones(n_dim) / n_dim

        # Track variances
        variances_unit_port = []
        variances_tilt = []

        for _ in range(n_iterations):
            # Step 1: Generate true correlation matrix
            true_cov_matrix = generate_true_cov_matrix(n_dim)

            for _ in range(5):
                # Step 2: Simulate data
                data = simulate_data(true_cov_matrix, n_data_samples)
                estimates = estimate_correlation(data)
                cov = estimates['']

                # Step 4: Compute portfolios
                w_unit_port, w_tilt = compute_portfolios(estimates['sample_corr'], n_samples=n_samples, port=port,
                                                         phi=phi)

                # Step 5: Compute true portfolio variances
                var_unit_port = compute_portfolio_variance(w_unit_port, true_cov_matrix)
                var_tilt = compute_portfolio_variance(w_tilt, true_cov_matrix)

                # Store the variances
                variances_unit_port.append(var_unit_port)
                variances_tilt.append(var_tilt)

                # Compute and compare the means
                mean_var_unit_port = np.mean(variances_unit_port)
                mean_var_tilt = np.mean(variances_tilt)

            print(f"Average Variance (Benchmark Port): {mean_var_unit_port}")
            print(f"Average Variance (Tilt Abil Port): {mean_var_tilt}")

import numpy as np
from collections import defaultdict


def leaderboard_comparison(ports: list, n_dim: int, n_data_samples: int,
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
    # Initial equal-weight portfolio
    w0 = np.ones(n_dim) / n_dim

    # Dictionary to track variances for each portfolio method
    variances = defaultdict(list)

    # Run through multiple iterations
    for iteration in range(n_iterations):
        # Step 1: Generate true covariance matrix
        true_cov_matrix = generate_true_cov_matrix(n_dim)

        # Perform multiple rounds of data simulation and portfolio evaluation
        for _ in range(5):  # Repeat multiple times per iteration
            # Step 2: Simulate data based on the true covariance matrix
            data = simulate_data(true_cov_matrix, n_data_samples)

            # Step 3: Estimate correlation matrices from the simulated data
            estimates = estimate_covariance(data)
            cov_est = estimates['ledoitwolf_cov']

            # Step 4: Evaluate each portfolio method
            for  portfolio_func in ports:
                method_name = portfolio_func.__name__.replace('_long_port','').replace('_port','')
                # Step 4a: Compute portfolio weights using the method
                w_portfolio = portfolio_func(cov=cov_est)

                # Step 5: Compute true portfolio variance using the true covariance matrix
                portfolio_variance = compute_portfolio_variance(w_portfolio, true_cov_matrix)

                # Store the variance for this method
                variances[method_name].append(portfolio_variance)

        # Periodically update and display the leaderboard
        if (iteration + 1) % update_interval == 0:
            update_leaderboard(variances, iteration + 1)


def update_leaderboard(variances: dict, iteration: int):
    """
    Update and print the leaderboard of portfolio methods based on their variances.

    :param variances: Dictionary of portfolio variances tracked by method name.
    :param iteration: Current iteration number (for display purposes).
    """
    print(f"\nLeaderboard after {iteration} iterations:")

    # Compute average variance for each portfolio method
    average_variances = {method: np.mean(var_list) for method, var_list in variances.items()}

    # Sort the portfolio methods by their average variance (ascending)
    sorted_methods = sorted(average_variances.items(), key=lambda x: x[1])

    # Display the sorted leaderboard
    max_method_name_length = max(len(method) for method in average_variances.keys())
    for rank, (method, avg_var) in enumerate(sorted_methods, 1):
        print(f"{rank:<3}. {method:<{max_method_name_length}} : Average Variance = {avg_var:>10.4f}")


if using_precise and using_scipy and using_sklearn:

    def ability_1k_100_port(cov):
        return ability_port(cov=cov, n_samples=1000, phi=1.0)

    def ability_100k_100_port(cov):
        return ability_port(cov=cov, n_samples=100000, phi=1.0)

    def ability_1k_50_port(cov):
        return ability_port(cov=cov, n_samples=1000, phi=0.5)

    def ability_100k_50_port(cov):
        return ability_port(cov=cov, n_samples=100000, phi=0.5)

    ability_ports = [ equal_long_port, ability_100k_50_port, ability_1k_100_port, ability_1k_50_port, ability_100k_50_port]
    conventional_ports = [unit_port, weak_long_port, unit_port_p050, hrp_diag_diag_s5_long_port ]
    ports = ability_ports + conventional_ports

    leaderboard_comparison(n_dim=250, ports=ports, n_data_samples=100)