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


def _independent_reference(u):
    """Independent Gaussian shares by one-dimensional quadrature."""
    from scipy.integrate import quad
    from scipy.stats import norm
    return np.array([
        quad(lambda x: norm.pdf(x - u[i]) * np.prod(norm.cdf(x - np.delete(u, i))),
             -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
        for i in range(len(u))])


def test_identity_covariance_is_an_independent_race():
    # issue #27: the contrast heuristic applied principal-factor analysis
    # to P Sigma P, whose idiosyncratic part P diag(D) P is not diagonal,
    # and invented a factor from the centring artefact (shares off by
    # 0.03-0.05 on this example); the projected fit returns V = 0
    u = np.array([0.6, -0.1, -0.5])
    got = shares(u, Sigma=np.eye(3), k=1)
    assert np.abs(got - _independent_reference(u)).max() < 1e-6
    # the identified object is P Sigma P; the loading orientation is a
    # gauge (V V' + diag D may differ from Sigma by 1 c' + c 1'), so the
    # representation is checked through the contrast covariance
    V, D = fit_factor_model(np.eye(3), 1)
    P = np.eye(3) - np.ones((3, 3)) / 3
    assert np.abs(P @ (V @ V.T + np.diag(D) - np.eye(3)) @ P).max() < 1e-8


def test_diagonal_covariances_keep_their_contrast_covariance_and_shares():
    rng = np.random.default_rng(3)
    n = 5
    u = rng.normal(size=n)
    d = rng.uniform(0.4, 2.5, n)
    P = np.eye(n) - np.ones((n, n)) / n
    for Sigma in (np.eye(n), np.diag(d)):
        for k in (1, 2):
            V, D = fit_factor_model(Sigma, k)
            fitted = V @ V.T + np.diag(D)
            assert np.abs(P @ (fitted - Sigma) @ P).max() < 1e-8
            p_sigma = shares(u, Sigma=Sigma, k=k)
            p_direct = shares(u, D=np.diag(Sigma))          # V = None: independent
            assert np.abs(p_sigma - p_direct).max() < 1e-6   # quadrature paths differ
            # a permutation of the alternatives permutes the shares
            perm = rng.permutation(n)
            p_perm = shares(u[perm], Sigma=Sigma[np.ix_(perm, perm)], k=k)
            assert np.abs(p_perm - p_sigma[perm]).max() < 1e-6


def test_sigma_path_forward_and_inverse_round_trip():
    u, V, D = _problem()
    Sigma = V @ V.T + np.diag(D)
    p = shares(u, Sigma=Sigma, k=2)
    u_hat = utilities_from_shares(p, Sigma=Sigma, k=2)
    assert np.abs(u_hat - (u - u.mean())).max() < 1e-4
    # the fitted representation reproduces the contrast covariance
    Vf, Df = fit_factor_model(Sigma, 2)
    P = np.eye(len(u)) - np.ones((len(u), len(u))) / len(u)
    assert np.abs(P @ (Vf @ Vf.T + np.diag(Df) - Sigma) @ P).max() < 1e-6
