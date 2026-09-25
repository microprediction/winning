"""A one-runner field has no covariance structure, and must not crash.

`race_probabilities([mu])` already returned `[1]`. Supplying the
equivalent `cov=[[v]]` crashed: the inner solver divides by
`(1 - 2/n) + n/n^2`, which is `-1 + 1 = 0` at `n = 1`. The `n <= 2`
branch added for #181 divides by the same quantity, so it covered `n = 2`
and left `n = 1` exactly where it was (#273).

R failed at the same field for a different reason -- it asks
`.factor_model_projected` for `min(k, n - 1) = 0` factors and gets a
`0 x 0` matrix back, which then fails to multiply.

There is nothing for a factor to correlate with one runner, so the whole
variance is idiosyncratic and the fit is exact at rank zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor import race_probabilities
from winning.factor.core import fit_covariance


def test_the_one_runner_race_is_certain_with_a_covariance():
    p = np.asarray(race_probabilities(np.array([2.0]),
                                      cov=np.array([[4.0]])))
    assert p.shape == (1,)
    assert abs(p[0] - 1.0) < 1e-12


def test_it_agrees_with_the_plain_one_runner_race():
    a = np.asarray(race_probabilities(np.array([2.0])))
    b = np.asarray(race_probabilities(np.array([2.0]), cov=np.array([[4.0]])))
    assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("v", [1e-6, 1.0, 4.0, 1e6])
def test_any_positive_variance_works(v):
    V, D, F, W = fit_covariance(np.array([[v]]))
    assert V.shape[0] == 1
    assert D.shape == (1,)
    assert abs(D[0] - v) < 1e-9 * max(v, 1.0)
    assert abs(float(np.sum(W)) - 1.0) < 1e-12


def test_the_report_has_the_keys_every_caller_reads():
    """`_fit_cov` reads rank, projected_residual_max, sharpness and
    contrast_residual_max. A short report raises KeyError several frames
    away from the cause."""
    *_, report = fit_covariance(np.array([[4.0]]), return_report=True)
    for key in ("rank", "projected_residual_max", "sharpness",
                "contrast_residual_max", "projected_residual_rel"):
        assert key in report, key
    assert report["rank"] == 0          # nothing for a factor to explain
    assert report["projected_residual_max"] == 0.0


@pytest.mark.parametrize("n", [2, 3, 5])
def test_larger_fields_are_untouched(n):
    C = np.eye(n) + 0.2 * (np.ones((n, n)) - np.eye(n))
    mu = np.linspace(0.0, 1.0, n)
    p = np.asarray(race_probabilities(mu, cov=C))
    assert p.shape == (n,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()
