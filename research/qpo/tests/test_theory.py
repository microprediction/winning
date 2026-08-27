"""Anchors for the sensitivity argument.

The claim being checked is the aggregate form of Plackett's relation: for ANY
symmetric perturbation of the covariance,

    d/dt p(Sigma + t Delta) |_0  =  (1/2) <Delta, H>  =  (1/2) sum_m lambda_m D^2_{u_m} p

with H the Hessian of p in the means and Delta = sum_m lambda_m u_m u_m'. The
one-half covers both cases at once: an off-diagonal entry appears twice in the
sum and carries no half of its own, a diagonal entry appears once and carries
the half.

Getting this tested honestly requires perturbations that stay exactly inside
the factor family, because that is the only family whose win probabilities can
be computed to machine accuracy. A first version of this file perturbed a
single covariance entry and then evaluated with a rank-(r+1) eigen-factor
model, which does not reproduce the perturbed matrix at all -- so the finite
difference measured the truncation, not the derivative. The constructions
below are exact at every t:

    Sigma(t) = V V' + t w w' + diag(d - t w^2)    (off-diagonal direction)
    Sigma(s) = V V' + diag(d + s e_i)             (diagonal direction)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics import tv_error  # noqa: E402
from pom import hermite_nodes, pom_fast  # noqa: E402
from theory import cov_perturbation_prediction, second_directional  # noqa: E402

POINTS = 2001


def _base(n=8, r=2, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n) * 0.5
    V = rng.standard_normal((n, r)) * 0.45
    d = rng.uniform(0.6, 1.2, n)
    return mu, V, d


def test_offdiagonal_direction_matches_the_hessian_form():
    """Delta = w w' - diag(w^2): zero diagonal, exactly representable."""
    n, r = 8, 2
    mu, V, d = _base(n, r, seed=1)
    rng = np.random.default_rng(11)
    w = rng.standard_normal(n) * 0.30
    Delta = np.outer(w, w) - np.diag(w ** 2)
    assert np.max(np.abs(np.diag(Delta))) < 1e-15

    t0, h = 0.5, 0.05

    def p_at(t):
        Vt = np.concatenate([V, np.sqrt(t) * w[:, None]], axis=1)
        dt = d - t * w ** 2
        assert dt.min() > 0
        F, W = hermite_nodes(r + 1, Q=17)
        return pom_fast(mu, Vt, dt, F, W, points=POINTS)

    actual = (p_at(t0 + h) - p_at(t0 - h)) / (2 * h)

    # predict at t0, so the base model is the one at t0
    Vt0 = np.concatenate([V, np.sqrt(t0) * w[:, None]], axis=1)
    dt0 = d - t0 * w ** 2
    F, W = hermite_nodes(r + 1, Q=17)
    pred, _, _ = cov_perturbation_prediction(mu, Vt0, dt0, F, W, Delta,
                                             n_dirs=n, points=POINTS)
    den = float(np.linalg.norm(actual))
    assert den > 1e-6
    rel = float(np.linalg.norm(pred - actual)) / den
    assert rel < 0.06, (rel, pred[:4], actual[:4])


def test_diagonal_direction_carries_the_half():
    """Delta = diag(e_i). If the half were missing this is off by a factor 2."""
    n, r = 8, 2
    mu, V, d = _base(n, r, seed=2)
    i = 3
    Delta = np.zeros((n, n))
    Delta[i, i] = 1.0
    h = 0.03
    F, W = hermite_nodes(r, Q=25)

    def p_at(s):
        return pom_fast(mu, V, d + s * np.eye(n)[i], F, W, points=POINTS)

    actual = (p_at(h) - p_at(-h)) / (2 * h)
    pred, _, _ = cov_perturbation_prediction(mu, V, d, F, W, Delta,
                                             n_dirs=n, points=POINTS)
    rel = float(np.linalg.norm(pred - actual)) / float(np.linalg.norm(actual))
    assert rel < 0.06, (rel,)
    # and the half is really there: without it the prediction doubles
    assert float(np.linalg.norm(2 * pred - actual)) / float(
        np.linalg.norm(actual)) > 0.5


def test_prediction_is_first_order_and_converges_as_the_step_shrinks():
    """Halving the perturbation must halve the relative prediction error."""
    n, r = 30, 2
    mu, V, d = _base(n, r, seed=4)
    rng = np.random.default_rng(5)
    w = rng.standard_normal(n) * 0.25
    Delta = np.outer(w, w) - np.diag(w ** 2)
    F, W = hermite_nodes(r, Q=21)
    p0 = pom_fast(mu, V, d, F, W, points=POINTS)
    pred, _, _ = cov_perturbation_prediction(mu, V, d, F, W, Delta,
                                             n_dirs=n, points=POINTS)

    errs = []
    for t in (0.4, 0.2, 0.1):
        Vt = np.concatenate([V, np.sqrt(t) * w[:, None]], axis=1)
        dt = d - t * w ** 2
        Ft, Wt = hermite_nodes(r + 1, Q=17)
        actual = pom_fast(mu, Vt, dt, Ft, Wt, points=POINTS) - p0
        errs.append(float(np.linalg.norm(actual - t * pred)) /
                    float(np.linalg.norm(actual)))
    # first-order error is O(t), so each halving of t should roughly halve it
    assert errs[1] < 0.7 * errs[0], errs
    assert errs[2] < 0.7 * errs[1], errs
    assert errs[2] < 0.12, errs
