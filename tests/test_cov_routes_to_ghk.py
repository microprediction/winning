"""race_probabilities(cov=) routes a degraded fit to scrambled-Sobol GHK.

Two failure classes were measured on 2026-09-21/22 and neither could be
fixed by fitting harder: a dense correlation the grammar fits badly
(dense n=16: 8.9e-3 off a 4M-path truth with the residual warnings
firing), and an exactly low-rank correlation the fit reproduces to ~1e-6
by going to full rank with D on its floor (n=8, rank 3: 6.6e-3 off its own
exact V/D answer, every residual check silent). qmc_ghk on the same
inputs: 4.7e-4 / 5.0e-4 at 1024 nodes, faster, deterministic, smooth.

So the forward normal race with no slopes -- the one case that needs no
factor form -- now goes to GHK when the fit is degraded. Everything that
needs the fit (slopes, a non-normal base, temperature, the inverse, the
prefix kernel) keeps it, with the warnings.
"""
import warnings

import numpy as np
import pytest

import winning.factor as wf
from winning.factor.core import fit_covariance


def _dense_corr(n, seed):
    A = np.random.default_rng(seed).normal(size=(n, n))
    S = A @ A.T + n * np.eye(n) * 0.3
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def _mc(mu, C, paths, seed=0):
    L = np.linalg.cholesky(C)
    r = np.random.default_rng(seed)
    wins = np.zeros(len(mu))
    done = 0
    while done < paths:
        m = min(400_000, paths - done)
        wins += np.bincount((mu + r.normal(size=(m, len(mu))) @ L.T).argmin(1), minlength=len(mu))
        done += m
    return wins / paths


def test_a_dense_cov_is_now_within_ghk_accuracy_of_truth():
    """Was 8.9e-3 through the fitted lattice."""
    C = _dense_corr(16, 16)
    mu = np.linspace(-0.5, 0.5, 16)
    truth = _mc(mu, C, 1_000_000)                      # se ~5e-4
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # routed: no warning
        p = np.asarray(wf.race_probabilities(mu, cov=C, points=257))
    assert np.abs(p - truth).max() < 2.0e-3
    assert abs(p.sum() - 1) < 1e-9


def test_a_healthy_low_rank_cov_still_uses_the_exact_lattice():
    """Exactly rank 3 at n=40: the fit stays rank 3 with D healthy, is not
    degraded, and cov= must equal the V=/D= lattice answer exactly."""
    rng = np.random.default_rng(40)
    Vt = rng.normal(size=(40, 3)) * 0.3
    D = 1.0 - (Vt ** 2).sum(1)
    C = Vt @ Vt.T + np.diag(D)
    mu = np.linspace(-0.5, 0.5, 40)
    Vf, Df, F, W, rep = fit_covariance(C, return_report=True)
    assert rep["rank"] < 40 and Df.min() > 1e-3 * np.diag(C).min()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p_cov = np.asarray(wf.race_probabilities(mu, cov=C, points=257))
        p_fit = np.asarray(wf.race_probabilities(mu, V=Vf, D=Df, F=F, W=W, points=257))
    assert np.abs(p_cov - p_fit).max() < 1e-12       # same lattice path; BLAS order may differ by an ulp


def test_the_routed_answer_is_deterministic_and_smooth():
    """Fixed seed: identical on repeat; a common shift of mu cannot move
    a race, and a small graded tilt moves it a little -- the properties a
    portfolio re-priced every period needs, which is why this is a
    quadrature-grade path and not a simulation."""
    C = _dense_corr(16, 16)
    mu = np.linspace(-0.5, 0.5, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.asarray(wf.race_probabilities(mu, cov=C, points=257))
        b = np.asarray(wf.race_probabilities(mu, cov=C, points=257))
        c = np.asarray(wf.race_probabilities(mu + 1e-3, cov=C, points=257))
        d = np.asarray(wf.race_probabilities(mu + 1e-3 * np.arange(16) / 16, cov=C, points=257))
    assert np.array_equal(a, b)
    assert np.abs(c - a).max() < 1e-12
    assert 0 < np.abs(d - a).max() < 2e-3


def test_calls_that_need_the_factor_form_keep_the_fit_and_warn():
    C = _dense_corr(16, 16)
    mu = np.linspace(-0.5, 0.5, 16)
    with pytest.warns(RuntimeWarning):
        p, sl = wf.race_probabilities(mu, cov=C, points=257, return_slopes=True)
    assert p.shape == sl.shape == (16,)
    with pytest.warns(RuntimeWarning):
        wf.race_probabilities(mu, cov=C, points=257, temperature=0.3)
    with pytest.warns(RuntimeWarning):
        wf.abilities_from_race(p, cov=C, points=257)
