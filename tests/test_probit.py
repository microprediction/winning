"""The probit door: max-wins semantics, reflection audited here only."""

import numpy as np

from winning.factor import hermite_nodes, win_probabilities_factor
from winning.probit import fit_factor_model, shares, utilities_from_shares


def _problem(n=30, k=2, seed=5):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n); u -= u.mean()
    V = rng.normal(0, 0.4, (n, k))
    D = rng.uniform(0.5, 1.5, n)
    return u, V, D


def test_reflection_is_exact():
    u, V, D = _problem()
    F, W = hermite_nodes(2)
    p = shares(u, V=V, D=D)
    q = win_probabilities_factor(-u, V, D, F, W)
    assert np.abs(p - q).max() == 0.0


def test_higher_utility_means_larger_share():
    u, V, D = _problem()
    p = shares(u, V=V, D=D)
    i, j = int(np.argmax(u)), int(np.argmin(u))
    assert p[i] > p[j]


def test_calibration_roundtrip_max_wins():
    u, V, D = _problem()
    p = shares(u, V=V, D=D)
    u_hat = utilities_from_shares(p, V=V, D=D)
    assert np.abs(u_hat - u).max() < 1e-4


def test_sigma_path_matches_explicit_fit():
    u, V, D = _problem()
    Sigma = V @ V.T + np.diag(D)
    Vf, Df = fit_factor_model(Sigma, 2)
    p_sigma, V_ret, D_ret = shares(u, Sigma=Sigma, k=2, return_fit=True)
    p_explicit = shares(u, V=Vf, D=Df)
    assert np.abs(p_sigma - p_explicit).max() == 0.0
    assert np.abs(V_ret - Vf).max() == 0.0


def test_zero_factor_default_is_independent_probit():
    u, _, _ = _problem()
    p = shares(u)                      # V=None -> independent, unit variance
    assert abs(p.sum() - 1) < 1e-12
    assert (np.argsort(p) == np.argsort(u)).all()
