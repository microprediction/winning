"""Prediction and likelihood choose the same quadrature.

`choice_loglik_and_score` gauge-centers `V` and dispatches on the
pairwise-safe bound `sqrt(2) max_i ||(PV)_i|| / sqrt(min D)`.
`MNProbit._prob_of` dispatched on `max_i ||V_i||` instead: no centering,
no `sqrt(2)`, no `D`. A fitted model could therefore optimise and report
under scrambled Sobol and then be PRICED on the 7-point Hermite tensor
(#213). Julia already matched its own likelihood.

Two ways it showed:

* not gauge-invariant -- only loading DIFFERENCES decide a race, yet
  adding a common offset of 5 to every row moved the statistic from 2.0
  to 7.0 and swapped 49 nodes for 1024 on an unchanged race;
* too weak -- a centred spread of 2.2 scored 2.2 against the
  likelihood's 3.11, so prediction took the Hermite tensor and priced
  2.9e-2 away from the rule the fit used.

The rule now lives in one place, `likelihood.sharpness_bound`, because it
was written twice and the copies drifted.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning import mnprobit as M
from winning.likelihood import nodes_for_likelihood, sharpness_bound

# centred spread 2.2: under the old prediction rule 2.2 (Hermite),
# under the likelihood's rule 3.11 (Sobol)
V_SHARP = np.array([[2.2], [-2.2], [0.5], [-0.5], [0.0]])


def _probs(mu, V):
    return np.column_stack([M._prob_of(mu, V, k) for k in range(mu.shape[1])])


def _mu(seed=0, T=150, J=5):
    return np.random.default_rng(seed).normal(size=(T, J)) * 0.5


@pytest.mark.parametrize("offset", [0.0, 5.0, -3.25])
def test_the_bound_is_gauge_invariant(offset):
    """A common offset cannot change a race, so it cannot change the
    node choice. The old statistic moved from 2.0 to 7.0."""
    assert abs(sharpness_bound(V_SHARP + offset)
               - sharpness_bound(V_SHARP)) < 1e-12


@pytest.mark.parametrize("offset", [5.0, -3.25])
def test_prediction_is_gauge_invariant(offset):
    mu = _mu()
    assert np.abs(_probs(mu, V_SHARP) - _probs(mu, V_SHARP + offset)).max() < 1e-12


def test_the_bound_carries_the_sqrt_two_and_the_idiosyncratic_scale():
    v = np.array([[1.0], [-1.0]])
    assert abs(sharpness_bound(v) - np.sqrt(2.0)) < 1e-12
    assert abs(sharpness_bound(v, D=np.array([0.25, 0.25]))
               - 2.0 * np.sqrt(2.0)) < 1e-12


def test_prediction_takes_the_same_node_set_as_the_likelihood():
    """The property that failed: both sides must escalate together."""
    sharp = sharpness_bound(V_SHARP)
    assert sharp > 3.0, "fixture must sit above the escalation threshold"
    weak = float(np.max(np.sqrt((V_SHARP ** 2).sum(axis=1))))
    assert weak < 3.0, "fixture must sit BELOW the old prediction rule"
    assert len(nodes_for_likelihood(1, 7, 7, sharp)[1]) > \
        len(nodes_for_likelihood(1, 7, 7, weak)[1])


def test_prediction_matches_the_sharper_quadrature_it_now_selects():
    from scipy.special import ndtr
    mu = _mu()
    F, W = nodes_for_likelihood(1, 7, 7, sharpness_bound(V_SHARP))
    Vc = V_SHARP - V_SHARP.mean(axis=0)
    Fq, zq = F[:, :1], F[:, 1]
    Vf = Fq @ Vc.T
    J = mu.shape[1]
    ref = np.zeros_like(mu)
    for k in range(J):
        rivals = [j for j in range(J) if j != k]
        for q in range(len(W)):
            m = mu + Vf[q][None, :]
            acc = np.ones(len(mu))
            for j in rivals:
                acc *= ndtr(m[:, k] + zq[q] - m[:, j])
            ref[:, k] += W[q] * acc
    assert np.abs(_probs(mu, V_SHARP) - ref).max() < 1e-12


def test_the_rows_are_a_distribution_on_the_sharp_field():
    """The weaker rule left them 1.6e-3 out; this is what that cost."""
    s = _probs(_mu(), V_SHARP).sum(axis=1)
    assert np.abs(s - 1.0).max() < 1e-3


def test_an_ordinary_field_is_unchanged():
    """Below the threshold both rules pick the Hermite tensor, so the
    answer is the same as before -- measured 3.161e-3 row-sum defect on
    main and 3.161e-3 here. The tolerance is what that tensor delivers
    on this field, not an aspiration: the SHARP field above lands at
    6.2e-4 precisely because it now escalates to Sobol."""
    V = np.array([[0.3], [-0.2], [0.1], [-0.1], [-0.1]])
    assert sharpness_bound(V) < 3.0
    p = _probs(_mu(seed=3), V)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 5e-3
    assert (p >= 0).all()
