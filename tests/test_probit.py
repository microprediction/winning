"""The probit door: max-wins semantics, reflection audited here only."""

import numpy as np

from winning.factor.core import qmc_nodes
from winning.factor.races import race_probabilities
from winning.probit import fit_factor_model, shares, utilities_from_shares


def _problem(n=30, k=2, seed=5):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n); u -= u.mean()
    V = rng.normal(0, 0.4, (n, k))
    D = rng.uniform(0.5, 1.5, n)
    return u, V, D


def test_reflection_is_exact():
    # the claim is that max-wins shares are the min-wins race read
    # backwards, against the same entry point the race layer offers;
    # pinning it to core with a hand-built node set instead pinned the
    # node rule, which is the race layer's to choose
    u, V, D = _problem()
    p = shares(u, V=V, D=D)
    q = race_probabilities(-u, V=V, D=D)
    # two ulp, not bit-exact: the two calls reach the same lattice
    # through different array objects. A path divergence is orders
    # larger, so the bound is still a real guard.
    assert np.abs(p - q).max() < 1e-15


def test_probit_inherits_the_adaptive_node_rule():
    # sharp loadings escalate the node family; a fixed Gauss-Hermite
    # rule here carried total variation 1.2e-3 against 2.8e-6
    rng = np.random.default_rng(5)
    n = 8
    mu = rng.normal(0, 1, n); mu -= mu.mean()
    V = rng.normal(0, 1.5, (n, 2)); D = np.ones(n)
    ref_F, ref_W = qmc_nodes(2, m=16)
    ref = race_probabilities(mu, V=V, D=D, F=ref_F, W=ref_W, points=501)
    p = shares(-mu, V=V, D=D, points=501)
    assert 0.5 * np.abs(p - ref).sum() < 1e-5


def test_probit_node_count_is_capped_in_rank():
    # the pruned tensor is 15**r without a budget: 24 s per call at
    # rank 6 against a capped 0.3 s
    import time
    rng = np.random.default_rng(6)
    n = 8
    mu = rng.normal(0, 1, n); mu -= mu.mean(); D = np.ones(n)
    V = rng.normal(0, 0.35, (n, 6))
    t0 = time.perf_counter()
    p = shares(-mu, V=V, D=D, points=257)
    assert time.perf_counter() - t0 < 5.0
    assert abs(p.sum() - 1.0) < 1e-8


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
    assert np.abs(p_sigma - p_explicit).max() < 1e-15
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
