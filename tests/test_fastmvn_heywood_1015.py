"""factorize_covariance refuses a Heywood solution (#1015).

The principal-factor iteration clips each residual variance at 1e-12.
On this well-conditioned 4x4 covariance (smallest eigenvalue 0.044) the
rank-3 fit wanted two negative residuals, held them at the floor, and
still reconstructed sigma to tolerance, so it was accepted as exact.
Two coordinates were then deterministic functions of the factors and
the factor quadrature integrated a step: the orthant came back 0.176791
against 0.175990 (scipy at abseps 1e-10; Monte Carlo 4e7 paths
0.175983 +- 0.000060, 13 SE below the old answer).
"""
import numpy as np

from winning.fastmvn import factorize_covariance, mvn_cdf_fast_info

C = np.array([[0.63, 0.29, 0.11, 0.52],
              [0.29, 0.86, 0.95, 0.16],
              [0.11, 0.95, 2.15, 1.28],
              [0.52, 0.16, 1.28, 2.45]])
TRUTH = 0.17598987098345426     # scipy, maxpts 1.6e6, abseps 1e-10


def test_heywood_fit_is_not_accepted_as_exact_1015():
    fd = factorize_covariance(C)
    if fd is not None:
        V, D = fd
        assert (D / np.diag(C)).min() > 1e-12
        assert np.abs(V @ V.T + np.diag(D) - C).max() < 1e-9


def test_heywood_covariance_is_priced_accurately_1015():
    p, method = mvn_cdf_fast_info(upper=np.zeros(4), sigma=C)
    assert abs(p - TRUTH) < 5e-5, (p, method)     # was 8.0e-4 off


def test_interior_exact_factor_model_still_takes_the_factor_path_1015():
    rng = np.random.default_rng(1015)
    V = rng.standard_normal((5, 2))
    D = rng.uniform(0.2, 1.0, 5)
    fd = factorize_covariance(V @ V.T + np.diag(D))
    assert fd is not None and (fd[1] / np.diag(V @ V.T + np.diag(D))).min() > 1e-12
    _, method = mvn_cdf_fast_info(upper=np.zeros(5), sigma=V @ V.T + np.diag(D))
    assert method.startswith("factor")
