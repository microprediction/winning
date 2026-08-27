"""Anchors. Nothing downstream is believed until these pass.

The point of each one:
  * N = 2 has a closed form, so the lattice kernel can be checked absolutely
    rather than against another approximation.
  * The kernel must agree with the deployed package function it is a retuned
    copy of -- the tight window and the max-wins convention are the only
    intended differences, so any disagreement is a bug in one of them.
  * Fast probit and factor Monte Carlo must agree under the SAME factor model,
    to Monte Carlo error. This is the phase III test in miniature: if it fails
    the probability calculation is wrong and nothing else is worth measuring.
  * Degenerate cases catch sign and normalisation errors that generic random
    tests sail past: equal means must give 1/N exactly, and a common factor
    must not move anything at all.
"""

import os
import sys

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factorize import contrast_factor, eig_factor  # noqa: E402
from metrics import qpo_efficiency, tanimoto_matrix, tv_error  # noqa: E402
from pom import (  # noqa: E402
    hermite_nodes, pom_alite, pom_factor_mc, pom_fast, pom_flite,
    pom_full_mc, pom_full_mc_scipy, pom_independent, sobol_nodes,
)


def _psd(n, rank, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, rank))
    return scale * (B @ B.T / rank + np.diag(rng.uniform(0.5, 1.5, n)))


# --------------------------------------------------------------------------
# closed form
# --------------------------------------------------------------------------

def test_two_candidates_closed_form():
    """Build the covariance from a factor model so the model is exactly right."""
    mu = np.array([0.3, -0.2])
    for V, d in (
        (np.array([[0.0], [0.0]]), np.array([1.0, 2.0])),
        (np.array([[0.9], [-0.4]]), np.array([0.6, 1.1])),
        (np.array([[0.9], [0.8]]), np.array([0.6, 1.1])),
    ):
        S = V @ V.T + np.diag(d)
        exact = ndtr((mu[0] - mu[1]) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]))
        F, W = hermite_nodes(1, Q=61)
        p = pom_fast(mu, V, d, F, W, points=4001)
        assert abs(p[0] - exact) < 5e-6, (V.ravel(), p, exact)


def test_independent_two_candidates():
    mu = np.array([0.3, -0.2])
    var = np.array([1.0, 2.0])
    exact = ndtr((mu[0] - mu[1]) / np.sqrt(var.sum()))
    p = pom_independent(mu, var, points=4001)
    assert abs(p[0] - exact) < 1e-7


# --------------------------------------------------------------------------
# agreement with the deployed package kernel
# --------------------------------------------------------------------------

def test_matches_winning_package_kernel():
    """Same model, same nodes: only the window and the wins-convention differ."""
    from winning.factor.core import win_probabilities_factor

    rng = np.random.default_rng(3)
    n, r = 40, 2
    mu = rng.standard_normal(n) * 0.5
    V = rng.standard_normal((n, r)) * 0.4
    d = rng.uniform(0.5, 1.5, n)
    F, W = hermite_nodes(r, Q=25)
    ours = pom_fast(mu, V, d, F, W, points=3001)
    # package is argmin: feed -mu, so its winner is our argmax
    theirs = win_probabilities_factor(-mu, V, d, F, W, points=3001,
                                      per_node_interval=True)
    assert np.max(np.abs(ours - theirs)) < 1e-9, np.max(np.abs(ours - theirs))


def test_adaptive_window_agrees_with_the_conservative_one():
    """The narrow window must change the cost, not the answer."""
    rng = np.random.default_rng(41)
    for n, spread in ((80, 0.3), (300, 1.0), (300, 6.0)):
        mu = rng.standard_normal(n) * spread
        d = rng.uniform(0.4, 1.6, n)
        p_safe = pom_independent(mu, d, points=20001, window="safe")
        p_adap = pom_independent(mu, d, points=20001, window="adaptive")
        assert np.max(np.abs(p_safe - p_adap)) < 1e-9, (n, spread,
                                                        np.max(np.abs(p_safe - p_adap)))


def test_adaptive_window_agrees_with_factor_model_too():
    rng = np.random.default_rng(42)
    n, r = 120, 2
    mu = rng.standard_normal(n) * 0.8
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.4, n)
    F, W = hermite_nodes(r, Q=15)
    a = pom_fast(mu, V, d, F, W, points=8001, window="safe")
    b = pom_fast(mu, V, d, F, W, points=8001, window="adaptive")
    assert np.max(np.abs(a - b)) < 1e-9, np.max(np.abs(a - b))


def test_window_is_tight_not_lossy():
    """Widely spread means: the tight window must not lose mass."""
    rng = np.random.default_rng(11)
    n = 60
    mu = rng.standard_normal(n) * 6.0        # spread over many sd
    d = rng.uniform(0.5, 1.5, n)
    p_tight = pom_independent(mu, d, points=4001)
    p_mc = pom_factor_mc(mu, None, d, M=400_000, seed=1)
    assert tv_error(p_tight, p_mc) < 0.004, tv_error(p_tight, p_mc)


# --------------------------------------------------------------------------
# fast probit vs factor Monte Carlo, same model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r", [1, 2, 3])
def test_fast_matches_factor_mc(r):
    rng = np.random.default_rng(20 + r)
    n = 30
    mu = rng.standard_normal(n) * 0.6
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.2, n)
    F, W = hermite_nodes(r, Q=21)
    p_fast = pom_fast(mu, V, d, F, W, points=3001)
    M = 2_000_000
    p_mc, se = pom_factor_mc(mu, V, d, M=M, seed=5, chunk=100_000, return_se=True)
    z = np.abs(p_fast - p_mc) / np.maximum(se, 1e-12)
    assert np.max(z) < 5.0, (np.max(z), np.argmax(z))


def test_sobol_nodes_converge_to_hermite():
    rng = np.random.default_rng(31)
    n, r = 25, 3
    mu = rng.standard_normal(n) * 0.6
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.2, n)
    F, W = hermite_nodes(r, Q=21)
    ref = pom_fast(mu, V, d, F, W, points=2001)
    prev = np.inf
    for m in (8, 12):
        Fs, Ws = sobol_nodes(r, m=m, seed=0)
        err = tv_error(ref, pom_fast(mu, V, d, Fs, Ws, points=2001))
        assert err < prev
        prev = err
    assert prev < 2e-3, prev


# --------------------------------------------------------------------------
# degenerate cases
# --------------------------------------------------------------------------

def test_equal_means_equal_variance_is_uniform():
    n = 50
    mu = np.zeros(n)
    d = np.full(n, 1.3)
    p = pom_independent(mu, d, points=2001)
    assert np.max(np.abs(p - 1.0 / n)) < 1e-12


def test_common_factor_is_choice_irrelevant():
    """Adding a factor that loads equally on everyone cannot move the argmax."""
    rng = np.random.default_rng(7)
    n = 40
    mu = rng.standard_normal(n) * 0.5
    d = rng.uniform(0.5, 1.5, n)
    V0 = np.zeros((n, 1))
    V1 = np.full((n, 1), 1.7)          # pure common mode
    F, W = hermite_nodes(1, Q=31)
    p0 = pom_fast(mu, V0, d, F, W, points=3001)
    p1 = pom_fast(mu, V1, d, F, W, points=3001)
    assert np.max(np.abs(p0 - p1)) < 1e-9, np.max(np.abs(p0 - p1))


def test_equicorrelated_is_uniform_under_equal_means():
    n = 30
    rho = 0.6
    S = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    V, d = eig_factor(S, 1)
    F, W = hermite_nodes(1, Q=41)
    p = pom_fast(np.zeros(n), V, d, F, W, points=3001)
    assert np.max(np.abs(p - 1.0 / n)) < 1e-10


# --------------------------------------------------------------------------
# factorization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("r", [0, 1, 4, 16])
def test_eig_factor_preserves_diagonal(r):
    S = _psd(40, 8, seed=2)
    V, d = eig_factor(S, r)
    recon = V @ V.T + np.diag(d)
    assert np.max(np.abs(np.diag(recon) - np.diag(S))) < 1e-12


def test_eig_factor_full_rank_reproduces_sigma():
    S = _psd(30, 30, seed=4)
    V, d = eig_factor(S, 30)
    assert np.max(np.abs(V @ V.T + np.diag(d) - S)) < 1e-10


def test_probabilities_depend_only_on_the_projected_covariance():
    """The theorem the quotient fit rests on: P Sigma P fixes the argmax law.

    Argmax probabilities are a functional of the difference vector, and
    e_i - e_j sums to zero, so anything Sigma does outside the mean-zero
    subspace is invisible. Adding a common loading column to V changes Sigma
    a great deal and must change nothing here.
    """
    rng = np.random.default_rng(19)
    n, r = 35, 2
    mu = rng.standard_normal(n) * 0.5
    V1 = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.2, n)
    c = rng.standard_normal(r) * 1.3
    V2 = V1 + np.ones((n, 1)) @ c[None, :]          # common loading added
    S1, S2 = V1 @ V1.T + np.diag(d), V2 @ V2.T + np.diag(d)
    assert np.linalg.norm(S1 - S2) > 1.0            # genuinely different
    from factorize import project
    assert np.linalg.norm(project(S1) - project(S2)) < 1e-10
    F, W = hermite_nodes(r, Q=25)
    p1 = pom_fast(mu, V1, d, F, W, points=3001)
    p2 = pom_fast(mu, V2, d, F, W, points=3001)
    assert np.max(np.abs(p1 - p2)) < 1e-9, np.max(np.abs(p1 - p2))


def test_contrast_factor_beats_eig_on_the_quotient_when_common_mode_is_strong():
    """Sigma = tau^2 11' + bb' + D: rank one must be spent on b, not on 1."""
    from factorize import project, quotient_cov_error
    rng = np.random.default_rng(9)
    n = 60
    b = rng.standard_normal(n)
    D = np.diag(rng.uniform(0.5, 1.5, n))
    S = 25.0 * np.ones((n, n)) + np.outer(b, b) + D

    err_e = quotient_cov_error(S, *eig_factor(S, 1))["quot_rel"]
    err_c = quotient_cov_error(S, *contrast_factor(S, 1))["quot_rel"]
    assert err_c < 0.5 * err_e, (err_c, err_e)
    # and the contrast model must be a legitimate (PSD) model
    Vc, dc = contrast_factor(S, 1)
    assert np.all(dc > 0)


def test_contrast_factor_probabilities_beat_eig_on_common_mode():
    """The covariance win has to show up in the probabilities too."""
    rng = np.random.default_rng(21)
    n = 45
    b = rng.standard_normal(n) * 1.0
    d0 = rng.uniform(0.5, 1.5, n)
    S = 25.0 * np.ones((n, n)) + np.outer(b, b) + np.diag(d0)
    mu = rng.standard_normal(n) * 0.6
    truth = pom_full_mc(mu, S, M=2_000_000, seed=3, chunk=100_000)
    F, W = hermite_nodes(1, Q=41)
    p_e = pom_fast(mu, *eig_factor(S, 1), F, W, points=3001)
    p_c = pom_fast(mu, *contrast_factor(S, 1), F, W, points=3001)
    e_c, e_e = tv_error(truth, p_c), tv_error(truth, p_e)
    # rank one cannot be exact here: P D P has O(1/n) off-diagonals no rank-one
    # term absorbs. The content of the test is the ratio, not the level.
    assert e_c < 0.35 * e_e, (e_c, e_e)
    assert e_c < 0.02, e_c


# --------------------------------------------------------------------------
# Monte Carlo reference
# --------------------------------------------------------------------------

def test_full_mc_matches_scipy_path():
    """Our Cholesky sampler and the authors' scipy sampler agree in law."""
    S = _psd(25, 5, seed=6, scale=0.3)
    mu = np.random.default_rng(6).standard_normal(25) * 0.3
    p1 = pom_full_mc(mu, S, M=400_000, seed=0, chunk=50_000)
    p2 = pom_full_mc_scipy(mu, S, M=400_000, seed=0)
    se = np.sqrt(np.maximum(p1 * (1 - p1), 1e-12) / 400_000)
    assert np.max(np.abs(p1 - p2) / (se * np.sqrt(2))) < 5.0


def test_rb_reference_matches_plain_mc():
    """Rao-Blackwellised reference is unbiased for the same quantity."""
    from pom import pom_full_rb
    S = _psd(40, 6, seed=22, scale=0.4)
    mu = np.random.default_rng(22).standard_normal(40) * 0.4
    p_mc, se_mc = pom_full_mc(mu, S, M=2_000_000, seed=1, chunk=100_000,
                              return_se=True)
    p_rb, se_rb = pom_full_rb(mu, S, M=20_000, seed=1, return_se=True)
    se = np.sqrt(se_mc ** 2 + se_rb ** 2)
    assert np.max(np.abs(p_rb - p_mc) / np.maximum(se, 1e-300)) < 5.0


def test_rb_reference_is_exact_on_a_factor_covariance():
    """Sigma = VV'+D: the RB sampler must reproduce the deterministic answer."""
    from pom import pom_full_rb
    rng = np.random.default_rng(23)
    n, r = 40, 2
    mu = rng.standard_normal(n) * 0.5
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.4, 1.0, n)
    S = V @ V.T + np.diag(d)
    F, W = hermite_nodes(r, Q=25)
    p_fast = pom_fast(mu, V, d, F, W, points=2001)
    p_rb, se = pom_full_rb(mu, S, M=20_000, seed=2, return_se=True)
    assert np.max(np.abs(p_fast - p_rb) / np.maximum(se, 1e-300)) < 5.0


def test_rb_reference_is_quieter_per_sample_but_not_per_second():
    """Records the measured trade, so nobody re-derives it from optimism.

    Conditioning on W buys a factor of a few in variance per draw -- how much
    depends on how large an isotropic part Sigma admits. It does NOT buy time:
    drawing a sample is a BLAS matmul, while the lattice pass is N*L log_ndtr
    evaluations, and log_ndtr costs orders of magnitude more per element than a
    BLAS flop. So the RB estimator is a cross-check on the reference, not the
    reference. On the qm9 posterior at N=1000 the measured numbers are 4.3x
    variance reduction per sample against roughly 400x slower per sample.
    """
    from pom import pom_full_rb
    S = _psd(200, 20, seed=24, scale=0.2)
    mu = np.random.default_rng(24).standard_normal(200) * 0.3
    _, se_rb = pom_full_rb(mu, S, M=4_000, seed=3, return_se=True)
    _, se_mc = pom_full_mc(mu, S, M=4_000, seed=3, chunk=4_000, return_se=True)
    assert se_rb.sum() < 0.6 * se_mc.sum(), (se_rb.sum(), se_mc.sum())


def test_full_mc_matches_fast_on_exact_factor_covariance():
    """Sigma built as VV'+D exactly: dense MC and fast probit must agree."""
    rng = np.random.default_rng(8)
    n, r = 30, 2
    mu = rng.standard_normal(n) * 0.5
    V = rng.standard_normal((n, r)) * 0.6
    d = rng.uniform(0.4, 1.0, n)
    S = V @ V.T + np.diag(d)
    F, W = hermite_nodes(r, Q=25)
    p_fast = pom_fast(mu, V, d, F, W, points=3001)
    p_mc, se = pom_full_mc(mu, S, M=2_000_000, seed=2, chunk=100_000, return_se=True)
    assert np.max(np.abs(p_fast - p_mc) / np.maximum(se, 1e-12)) < 5.0


# --------------------------------------------------------------------------
# LITE
# --------------------------------------------------------------------------

def test_flite_normalises_and_is_monotone():
    rng = np.random.default_rng(12)
    n = 500
    mu = rng.standard_normal(n)
    var = rng.uniform(0.5, 2.0, n)
    p = pom_flite(mu, var)
    assert abs(p.sum() - 1) < 1e-10
    # at equal variance, F-LITE must rank exactly by mean
    p2 = pom_flite(mu, np.full(n, 1.0))
    assert np.all(np.argsort(-p2) == np.argsort(-mu))


def test_flite_matches_reference_implementation():
    """numpy transcription against the authors' jax flite.py."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0, os.path.expanduser("~/github/LITE"))
    from flite import flite_pom

    rng = np.random.default_rng(13)
    for n in (50, 500):
        mu = rng.standard_normal(n) * 0.4
        var = rng.uniform(0.3, 1.7, n)
        ours = pom_flite(mu, var)
        theirs = np.asarray(jax.device_get(
            flite_pom(jnp.asarray(mu, dtype=jnp.float64),
                      jnp.asarray(np.sqrt(var), dtype=jnp.float64))))
        assert np.max(np.abs(ours - theirs)) < 1e-6, np.max(np.abs(ours - theirs))


def test_flite_and_alite_approximate_the_independent_truth():
    """Both are approximations to pom_independent; they should be close to it."""
    rng = np.random.default_rng(14)
    n = 400
    mu = rng.standard_normal(n) * 0.3
    var = rng.uniform(0.5, 1.5, n)
    exact = pom_independent(mu, var, points=4001)
    assert tv_error(exact, pom_flite(mu, var)) < 0.15
    assert tv_error(exact, pom_alite(mu, var)) < 0.15


def test_alite_normalises():
    rng = np.random.default_rng(15)
    n = 200
    p = pom_alite(rng.standard_normal(n) * 0.3, rng.uniform(0.5, 1.5, n))
    assert abs(p.sum() - 1) < 1e-9
    assert np.all(p >= 0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_efficiency_is_one_for_identical_scores():
    rng = np.random.default_rng(16)
    p = rng.dirichlet(np.ones(300))
    assert abs(qpo_efficiency(p, p, 100) - 1.0) < 1e-12


def test_efficiency_bounded_and_sensible():
    rng = np.random.default_rng(17)
    p = rng.dirichlet(np.ones(300))
    q = rng.permutation(p)
    e = qpo_efficiency(p, q, 100)
    assert 0.0 <= e <= 1.0 + 1e-12


def test_tanimoto_matches_qpo_utils():
    rng = np.random.default_rng(18)
    X = rng.integers(0, 4, size=(12, 30)).astype(float)
    T = tanimoto_matrix(X)
    # reference: the authors' pairwise loop
    n = len(X)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot = X[i].dot(X[j])
            R[i, j] = dot / (np.sum(X[i] ** 2) + np.sum(X[j] ** 2) - dot)
    assert np.max(np.abs(T - R)) < 1e-12
    assert np.max(np.abs(np.diag(T) - 1.0)) < 1e-12
