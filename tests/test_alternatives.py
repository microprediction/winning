"""The alternatives module agrees with the engine on what it computes."""
import numpy as np
import pytest

from winning.alternatives import reduced_rank_representation
from winning.factor import race_probabilities


def _instance(n, k, seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    V = rng.normal(0, 0.6, (n, k))
    d = 0.3 + rng.random(n)
    return mu, V, d


def test_reduced_rank_matches_engine():
    """P(i wins) from the Marsaglia rectangle equals the engine's
    max-wins share, winner by winner, via scipy's MVN CDF at small n."""
    from scipy.stats import multivariate_normal
    mu, V, d = _instance(6, 2, 3)
    p_engine = race_probabilities(-mu, V=-V, D=d)
    for i in range(6):
        r = reduced_rank_representation(mu, V, d, i)
        cov = r["B"] @ r["B"].T + np.diag(r["D_minus"])
        p_i = multivariate_normal(mean=np.zeros(5), cov=cov,
                                  allow_singular=True).cdf(r["upper"])
        assert abs(p_i - p_engine[i]) < 2e-3


def test_reduced_rank_covariance_identity():
    """B B' + diag(D_minus) equals the difference covariance built
    from the full Sigma."""
    mu, V, d = _instance(8, 3, 4)
    Sigma = V @ V.T + np.diag(d)
    i = 2
    A = np.delete(np.eye(8), i, axis=0)
    A[:, i] = -1.0
    r = reduced_rank_representation(mu, V, d, i)
    lhs = r["B"] @ r["B"].T + np.diag(r["D_minus"])
    rhs = A @ Sigma @ A.T
    assert np.abs(lhs - rhs).max() < 1e-12


def test_cdf_gradient_matches_engine():
    jax = pytest.importorskip("jax")            # noqa: F841
    from winning.alternatives import cdf_gradient_shares
    mu, V, d = _instance(30, 2, 5)
    p_engine = race_probabilities(-mu, V=-V, D=d)
    p_alt = cdf_gradient_shares(mu, V, d, n_samples=256, n_grid=64)
    tv = 0.5 * np.abs(p_alt - p_engine).sum()
    assert tv < 0.03
    assert abs(p_alt.sum() - 1.0) < 1e-6


def test_per_winner_shares_match_engine():
    from winning.alternatives import per_winner_reduced_rank_shares
    mu, V, d = _instance(40, 2, 6)
    p_engine = race_probabilities(-mu, V=-V, D=d)
    p = per_winner_reduced_rank_shares(mu, V, d, n_samples=2048)
    assert 0.5 * np.abs(p - p_engine).sum() < 0.01
