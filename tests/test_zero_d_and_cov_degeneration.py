"""Two edge cases found while measuring cov= against Monte Carlo (2026-09-21).

1. A D entry of exactly zero reached the lattice and died as
   `OverflowError: cannot convert float infinity to integer` in
   forward_grid's grid sizing (it divides by the smallest sd). A zero
   variance is legal as a BELIEF (as_variance: a perfectly known
   quantity) but not as performance noise on the lattice, so the lattice
   kernels now ask as_idio for strict positivity and raise a ValueError
   that says so.

2. The cov= warnings judge how well V V' + D reproduces cov. On dense
   correlations at n = 16-30 they were silent (fit residual ~1e-6) while
   the probabilities were 2-9e-3 off a 2M-path truth. What predicted the
   error was the fit reaching full rank or binding the D floor -- the
   conditional race is then a near-step the factor nodes cannot resolve.
   cov= now warns on that signal and names the alternative
   (winning.methods.qmc_ghk: ~5e-4 in a tenth of the time).
"""
import warnings

import numpy as np
import pytest

import winning.factor as wf
from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes, jacobian_vector_product,
                                 win_probabilities_factor)
from winning.shapes import as_idio, as_variance

N = 5
MU = np.linspace(-0.4, 0.4, N)
V = np.array([[0.6], [-0.2], [0.3], [0.1], [-0.5]])
D_ZERO = np.array([0.9, 0.0, 1.1, 0.8, 1.0])
D_TINY = np.array([0.9, 1e-9, 1.1, 0.8, 1.0])


def test_zero_d_raises_a_named_error_not_an_overflow():
    with pytest.raises(ValueError, match=r"strictly positive.*point mass"):
        wf.race_probabilities(MU, V=V, D=D_ZERO, points=257)


def test_every_lattice_kernel_rejects_zero_d():
    F, W = hermite_nodes(1, Q=11)
    with pytest.raises(ValueError, match=r"strictly positive"):
        win_probabilities_factor(MU, V, D_ZERO, F, W)
    with pytest.raises(ValueError, match=r"strictly positive"):
        jacobian_vector_product(MU, V, D_ZERO, F, W, np.ones(N))
    p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    with pytest.raises(ValueError, match=r"strictly positive"):
        abilities_from_probabilities_factor(p, V, D_ZERO, F, W)


def test_a_tiny_positive_d_is_still_accepted():
    p = np.asarray(wf.race_probabilities(MU, V=V, D=D_TINY, points=257))
    assert abs(p.sum() - 1) < 1e-9


def test_zero_is_still_a_legal_belief_variance():
    """as_variance (ratings beliefs) keeps accepting zero: perfectly known."""
    out = as_variance(np.array([0.0, 0.5, 0.0]), 3)
    assert out.min() == 0.0
    assert as_idio(np.array([0.0, 0.5]), 2).min() == 0.0      # default: lenient
    with pytest.raises(ValueError, match=r"strictly positive"):
        as_idio(np.array([0.0, 0.5]), 2, positive=True)


def _dense_corr(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    S = A @ A.T + n * np.eye(n) * 0.3
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def _exact_factor_corr(n, r, seed):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, r)) * 0.35
    V = V / np.sqrt((V ** 2).sum(1, keepdims=True)) * np.sqrt(0.6) * rng.random((n, 1)) ** 0.3
    return V @ V.T + np.diag(1.0 - (V ** 2).sum(1))


def test_cov_warns_when_the_fit_degenerates():
    """Exactly 3-factor at n=8 (#118): the fit goes to full rank with D on
    the floor, reproduces cov to ~1e-6, and prices 6.6e-3 off the exact
    V/D answer -- the case the residual checks cannot see."""
    C = _exact_factor_corr(8, 3, seed=8)
    mu = np.linspace(-0.5, 0.5, 8)
    with pytest.warns(RuntimeWarning, match=r"cov= fit degenerated.*qmc_ghk"):
        wf.race_probabilities(mu, cov=C, points=257)


def test_a_badly_fitted_dense_cov_is_caught_by_the_residual_check():
    """The other case IS covered, by the existing warning: dense n=16 has
    projected residual 8.3e-2 > 0.05. Pinned so the two warnings stay
    complementary rather than one quietly replacing the other."""
    C = _dense_corr(16, 16)
    mu = np.linspace(-0.5, 0.5, 16)
    with pytest.warns(RuntimeWarning, match=r"imperfectly served|nearly singular"):
        wf.race_probabilities(mu, cov=C, points=257)


def test_cov_does_not_warn_on_a_healthy_low_rank_fit():
    """Exactly rank-3 at n=40: the fit stays rank 3 with D at 0.16 of the
    diagonal (probed), so the degeneration warning must stay quiet."""
    rng = np.random.default_rng(40)
    Vt = rng.normal(size=(40, 3)) * 0.3
    C = Vt @ Vt.T + np.diag(1.0 - (Vt ** 2).sum(1))
    mu = np.linspace(-0.5, 0.5, 40)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wf.race_probabilities(mu, cov=C, points=257)
    assert not [x for x in w if "degenerated" in str(x.message)]
