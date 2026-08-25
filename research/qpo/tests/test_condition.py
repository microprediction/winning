"""Anchors for the rank-one posterior downdate.

The update is exact arithmetic, so it is checked against the thing it
replaces: form Sigma densely, do the textbook Gaussian conditioning, and
compare. The distinction the tests enforce is the one that matters --
dropping the observed candidate is EXACT at the same rank with the diagonal
untouched, while keeping it is only exact when the observation is noiseless.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from condition import condition_drop, condition_keep, sigma_column  # noqa: E402
from pom import hermite_nodes, pom_fast, pom_full_mc  # noqa: E402


def _model(n=25, r=3, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n) * 0.5
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.2, n)
    return mu, V, d


def _dense_update(mu, Sigma, k, y, noise):
    col = Sigma[:, k]
    s = Sigma[k, k] + noise
    return mu + (y - mu[k]) * col / s, Sigma - np.outer(col, col) / s


def test_sigma_column_is_the_column():
    mu, V, d = _model(seed=1)
    S = V @ V.T + np.diag(d)
    for k in (0, 7, 24):
        assert np.max(np.abs(sigma_column(V, d, k) - S[:, k])) < 1e-13


@pytest.mark.parametrize("noise", [0.0, 0.3, 2.0])
@pytest.mark.parametrize("seed", [2, 3])
def test_drop_update_is_exact_at_the_same_rank(noise, seed):
    """The claim: exact, rank preserved, diagonal untouched, any noise level."""
    mu, V, d = _model(n=30, r=4, seed=seed)
    S = V @ V.T + np.diag(d)
    k, y = 5, 0.8
    mu_ref, S_ref = _dense_update(mu, S, k, y, noise)
    keep = np.arange(len(mu)) != k

    mu_new, V_new, d_new = condition_drop(mu, V, d, k, y, noise)
    assert V_new.shape == (len(mu) - 1, V.shape[1])          # rank preserved
    assert np.max(np.abs(d_new - d[keep])) == 0.0            # diagonal untouched
    assert np.max(np.abs(mu_new - mu_ref[keep])) < 1e-12
    S_new = V_new @ V_new.T + np.diag(d_new)
    assert np.max(np.abs(S_new - S_ref[np.ix_(keep, keep)])) < 1e-11


def test_keep_update_is_exact_only_without_noise():
    """Records the limitation rather than hiding it."""
    mu, V, d = _model(n=25, r=3, seed=4)
    S = V @ V.T + np.diag(d)
    k, y = 5, 0.8
    for noise, tol, exact in ((0.0, 1e-9, True), (0.3, 1e-9, False)):
        mu_ref, S_ref = _dense_update(mu, S, k, y, noise)
        mu_new, V_new, d_new = condition_keep(mu, V, d, k, y, noise,
                                              rank=V.shape[1] + 1)
        err = np.max(np.abs(V_new @ V_new.T + np.diag(d_new) - S_ref))
        assert np.max(np.abs(mu_new - mu_ref)) < 1e-12
        assert bool(err < tol) is exact, (noise, err)   # np.bool_ is not bool
        # in both cases the marginal variances are reproduced
        recon = np.sum(V_new ** 2, axis=1) + d_new
        assert np.max(np.abs(recon - np.diag(S_ref))) < 1e-9


def test_observed_candidate_is_pinned_when_noiseless():
    mu, V, d = _model(n=30, r=3, seed=5)
    k = 9
    mu_new, V_new, d_new = condition_keep(mu, V, d, k, 0.2, noise=0.0,
                                          rank=V.shape[1] + 1)
    assert float(np.sum(V_new[k] ** 2) + d_new[k]) < 1e-9
    assert abs(mu_new[k] - 0.2) < 1e-12


def test_a_high_observation_lifts_its_neighbours():
    """Sanity on direction: a high value at k must raise correlated candidates."""
    mu, V, d = _model(n=40, r=2, seed=6)
    F, W = hermite_nodes(2, Q=21)
    p0 = pom_fast(mu, V, d, F, W, points=1001)
    k = 0
    corr = V @ V[k]
    friend = int(np.argmax(np.where(np.arange(40) == k, -np.inf, corr)))
    mu_hi, V_hi, d_hi = condition_drop(mu, V, d, k, mu[k] + 4.0, noise=0.1)
    p_hi = pom_fast(mu_hi, V_hi, d_hi, F, W, points=1001)
    idx = np.arange(40)[np.arange(40) != k]
    j = int(np.flatnonzero(idx == friend)[0])
    assert p_hi[j] > p0[friend]


def test_argmax_distribution_after_update_matches_dense_monte_carlo():
    """End to end: the conditioned factor model must give the right p."""
    mu, V, d = _model(n=30, r=2, seed=7)
    S = V @ V.T + np.diag(d)
    k, y, noise = 3, 0.9, 0.2
    mu_ref, S_ref = _dense_update(mu, S, k, y, noise)
    keep = np.arange(len(mu)) != k
    mu_new, V_new, d_new = condition_drop(mu, V, d, k, y, noise)
    F, W = hermite_nodes(2, Q=21)
    p_fast = pom_fast(mu_new, V_new, d_new, F, W, points=1001)
    p_mc, se = pom_full_mc(mu_ref[keep], S_ref[np.ix_(keep, keep)],
                           M=2_000_000, seed=1, chunk=100_000, return_se=True)
    assert np.max(np.abs(p_fast - p_mc) / np.maximum(se, 1e-12)) < 5.0
