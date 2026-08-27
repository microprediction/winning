"""Anchors for the one-factor conditional envelope.

The order matters. The envelope has to be right before the conditional shares
mean anything; the conditional shares have to be right before the estimator
means anything; and the analytic Jacobian has to match finite differences of
the estimator taken with THE SAME residual draws before any Newton step is
attempted. That last one is the check the brief calls mandatory.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from envelope import (apply_laplacian, conditional_edges,  # noqa: E402
                      conditional_shares, laplacian_from_edges, project,
                      raw_shares, rb_shares, split_one_factor, upper_envelope)


# --------------------------------------------------------------------------
# the envelope itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(5, 0), (20, 1), (200, 2)])
def test_envelope_matches_brute_force_on_a_grid(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    idx, tau = upper_envelope(b, c)
    z = np.linspace(-8, 8, 20001)
    brute = np.argmax(c[:, None] + b[:, None] * z[None, :], axis=0)
    # the envelope must name the same winner at every grid point
    lo = np.concatenate([[-np.inf], tau])
    hi = np.concatenate([tau, [np.inf]])
    seg = np.searchsorted(tau, z, side="right")
    ours = idx[seg]
    assert np.mean(ours == brute) > 0.9995, np.mean(ours == brute)


def test_envelope_breakpoints_increase_and_lines_are_ordered():
    rng = np.random.default_rng(3)
    b, c = rng.standard_normal(100), rng.standard_normal(100)
    idx, tau = upper_envelope(b, c)
    assert np.all(np.diff(tau) > 0)
    assert np.all(np.diff(b[idx]) > 0)      # envelope is ordered by slope


def test_equal_slopes_keep_only_the_largest_intercept():
    b = np.array([1.0, 1.0, 2.0])
    c = np.array([0.0, 5.0, 0.0])
    idx, tau = upper_envelope(b, c)
    assert 0 not in idx and 1 in idx


def test_single_line_and_two_lines():
    idx, tau = upper_envelope(np.array([1.0]), np.array([0.3]))
    assert list(idx) == [0] and len(tau) == 0
    idx, tau = upper_envelope(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert len(idx) == 2 and len(tau) == 1
    assert abs(tau[0] - 1.0) < 1e-12       # 1 + 0z = 0 + 1z at z = 1


# --------------------------------------------------------------------------
# conditional shares
# --------------------------------------------------------------------------

def test_conditional_shares_sum_to_one_and_are_nonnegative():
    rng = np.random.default_rng(4)
    for _ in range(20):
        n = int(rng.integers(2, 80))
        b, c = rng.standard_normal(n), rng.standard_normal(n)
        q, idx, tau = conditional_shares(b, c)
        assert abs(q.sum() - 1.0) < 1e-12
        assert np.all(q >= 0)
        assert np.all(q[np.setdiff1d(np.arange(n), idx)] == 0)


def test_conditional_shares_match_direct_simulation_of_z():
    """Given eta, simulate Z and count winners: must match the exact segments."""
    rng = np.random.default_rng(5)
    n = 30
    b, c = rng.standard_normal(n), rng.standard_normal(n) * 0.6
    q, _, _ = conditional_shares(b, c)
    z = rng.standard_normal(400_000)
    win = np.argmax(c[:, None] + np.outer(b, z), axis=0)
    emp = np.bincount(win, minlength=n) / len(z)
    se = np.sqrt(np.maximum(emp * (1 - emp), 1e-12) / len(z))
    assert np.max(np.abs(q - emp) / np.maximum(se, 1e-12)) < 5.0


# --------------------------------------------------------------------------
# the estimator is unbiased for the right thing
# --------------------------------------------------------------------------

def test_rb_estimator_agrees_with_raw_winner_counting():
    rng = np.random.default_rng(6)
    n = 40
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.5 * np.eye(n))
    mu = rng.standard_normal(n) * 0.5
    mu -= mu.mean()
    b, R, A, _ = split_one_factor(Sigma)

    w, U = np.linalg.eigh(Sigma)
    S_sqrt = U * np.sqrt(np.maximum(w, 0))
    p_raw, se_raw = raw_shares(mu, S_sqrt, M=400_000, seed=1)
    p_rb, se_rb = rb_shares(mu, b, A, M=4000, seed=2)
    se = np.sqrt(se_raw ** 2 + se_rb ** 2)
    assert np.max(np.abs(p_raw - p_rb) / np.maximum(se, 1e-12)) < 5.0


# --------------------------------------------------------------------------
# the Jacobian, against finite differences with common random numbers
# --------------------------------------------------------------------------

def test_jacobian_matches_finite_differences():
    """The mandatory check: same residual draws on both sides of the difference.

    The discrepancy falls as O(eps), not O(eps^2), and that is a property of
    the estimator rather than a defect. With the draws held fixed, the sample
    average is piecewise smooth: when a line enters or leaves the envelope its
    conditional share leaves zero with a strictly positive derivative, so the
    finite-sample map has a kink there. A fraction of draws of order eps has
    such a kink inside the differencing interval, each contributing O(1), so
    the error is first order. Taking the expectation over eta smooths this
    away, which is why the analytic Laplacian is still the derivative of the
    quantity being estimated.
    """
    rng = np.random.default_rng(7)
    n, M = 25, 200
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.4 * np.eye(n))
    mu = rng.standard_normal(n) * 0.4
    mu -= mu.mean()
    b, R, A, _ = split_one_factor(Sigma)

    p, _, edges, w = rb_shares(mu, b, A, M=M, seed=11, want_jacobian=True)
    v = rng.standard_normal(n)
    v -= v.mean()
    Jv = apply_laplacian(edges, w, v)

    rels = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        pp, _ = rb_shares(mu + eps * v, b, A, M=M, seed=11)
        pm, _ = rb_shares(mu - eps * v, b, A, M=M, seed=11)
        fd = (pp - pm) / (2 * eps)
        rels.append(np.linalg.norm(Jv - fd) / max(np.linalg.norm(fd), 1e-300))
    # decreases, then flattens at numerical precision
    assert rels[1] < 0.5 * rels[0], rels
    assert rels[2] < 0.5 * rels[1], rels
    assert min(rels) < 5e-3, rels


def test_jacobian_is_a_psd_laplacian():
    rng = np.random.default_rng(8)
    n, M = 20, 128
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.4 * np.eye(n))
    mu = rng.standard_normal(n) * 0.4
    b, R, A, _ = split_one_factor(Sigma)
    _, _, edges, w = rb_shares(mu, b, A, M=M, seed=12, want_jacobian=True)
    J = laplacian_from_edges(edges, w, n)
    assert np.max(np.abs(J @ np.ones(n))) < 1e-12
    assert np.max(np.abs(J - J.T)) < 1e-14
    ev = np.linalg.eigvalsh(J)
    assert ev.min() > -1e-12
    assert np.all(w > 0)


def test_conditional_jacobian_is_a_path_graph():
    """Conditional on eta only adjacent envelope segments touch: at most N-1 edges."""
    rng = np.random.default_rng(9)
    n = 60
    b, c = rng.standard_normal(n), rng.standard_normal(n)
    q, idx, tau = conditional_shares(b, c)
    edges, w = conditional_edges(b, idx, tau)
    assert len(edges) == len(idx) - 1
    assert len(edges) <= n - 1
    # consecutive edges chain together
    assert np.all(edges[1:, 0] == edges[:-1, 1])


# --------------------------------------------------------------------------
# the point of the exercise
# --------------------------------------------------------------------------

def test_rao_blackwell_reduces_variance():
    """Per draw, conditioning on eta must beat counting winners."""
    rng = np.random.default_rng(10)
    n = 50
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.5 * np.eye(n))
    mu = rng.standard_normal(n) * 0.5
    b, R, A, _ = split_one_factor(Sigma)
    w, U = np.linalg.eigh(Sigma)
    S_sqrt = U * np.sqrt(np.maximum(w, 0))

    M = 2000
    _, se_rb = rb_shares(mu, b, A, M=M, seed=20)
    _, se_raw = raw_shares(mu, S_sqrt, M=M, seed=20)
    assert se_rb.sum() < se_raw.sum(), (se_rb.sum(), se_raw.sum())


# --------------------------------------------------------------------------
# the vectorised path must agree with the scalar one it replaces
# --------------------------------------------------------------------------

def test_batch_envelope_matches_the_scalar_envelope():
    from envelope import conditional_shares_batch
    rng = np.random.default_rng(31)
    n, M = 60, 40
    b = rng.standard_normal(n)
    order = np.argsort(b, kind="stable")
    bs = b[order]
    Cfull = rng.standard_normal((M, n))
    q_batch, _, _, _ = conditional_shares_batch(bs, Cfull[:, order], order=order,
                                                n_total=n)
    for t in range(M):
        q_ref, _, _ = conditional_shares(b, Cfull[t])
        assert np.max(np.abs(q_batch[t] - q_ref)) < 1e-12, t


def test_batch_estimator_matches_the_scalar_estimator():
    from envelope import rb_shares_batch
    rng = np.random.default_rng(32)
    n = 40
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.5 * np.eye(n))
    mu = rng.standard_normal(n) * 0.5
    b, R, A, _ = split_one_factor(Sigma)
    p1, s1 = rb_shares(mu, b, A, M=256, seed=5)
    p2, s2 = rb_shares_batch(mu, b, A, M=256, seed=5)
    assert np.max(np.abs(p1 - p2)) < 1e-12
    assert np.max(np.abs(s1 - s2)) < 1e-12


def test_batch_jacobian_matches_finite_differences():
    from envelope import rb_shares_batch
    rng = np.random.default_rng(33)
    n, M = 30, 400
    A0 = rng.standard_normal((n, n))
    Sigma = project(A0 @ A0.T / n + 0.4 * np.eye(n))
    mu = rng.standard_normal(n) * 0.4
    mu -= mu.mean()
    b, R, A, _ = split_one_factor(Sigma)
    p, _, edges, w = rb_shares_batch(mu, b, A, M=M, seed=21, want_jacobian=True)
    v = rng.standard_normal(n)
    v -= v.mean()
    Jv = apply_laplacian(edges, w, v)
    eps = 1e-5
    pp, _ = rb_shares_batch(mu + eps * v, b, A, M=M, seed=21)
    pm, _ = rb_shares_batch(mu - eps * v, b, A, M=M, seed=21)
    fd = (pp - pm) / (2 * eps)
    rel = np.linalg.norm(Jv - fd) / max(np.linalg.norm(fd), 1e-300)
    assert rel < 5e-3, rel
    J = laplacian_from_edges(edges, w, n)
    assert np.max(np.abs(J @ np.ones(n))) < 1e-12
    assert np.linalg.eigvalsh(J).min() > -1e-12
