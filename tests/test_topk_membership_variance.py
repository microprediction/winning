"""Top-k membership can raise a rival's posterior variance.

The event "entrant 0 finishes in the top two of three" is the union
{X0 > X1} or {X0 > X2}. A union of half-spaces is not convex, so the
event probability is not log-concave in the prior means and the
curvature d^2 log P / d m_j^2 can be positive. With unit skill and
noise variances and equal means, P = 2/3 and

    Var(s_0 | E) = 1 - (4 sqrt 3 + 9) / (32 pi)   = 0.8415592...
    Var(s_j | E) = 1 + (8 sqrt 3 - 9) / (128 pi)  = 1.0120768...,  j = 1, 2.

The second case moves the named entrant's mean to -4. The reference
there is one-dimensional adaptive quadrature of
P(E | X_j = x) = 1 - P(X_0 < x, X_0 < X_k) against the rival's
predictive density, followed by Gaussian regression of skill on
performance.

A winner (k = 1) is an intersection of half-spaces, where the
probability is log-concave and no variance can rise.
"""
import math

import numpy as np

from winning.ratings.nway import update_topk_correlated, update_winner_correlated


def test_equal_means_matches_closed_form():
    m, v, logZ = update_topk_correlated(np.zeros(3), np.ones(3), member=0, k=2, beta2=1.0)
    np.testing.assert_allclose(math.exp(logZ), 2.0 / 3.0, atol=1e-8)
    member = 1.0 - (4.0 * math.sqrt(3.0) + 9.0) / (32.0 * math.pi)
    rival = 1.0 + (8.0 * math.sqrt(3.0) - 9.0) / (128.0 * math.pi)
    np.testing.assert_allclose(v, [member, rival, rival], atol=1e-6)
    assert v[1] > 1.0 and v[2] > 1.0


def test_weak_member_raises_rival_variance_by_a_quarter():
    m0 = np.array([-4.0, 0.0, 0.0])
    m, v, logZ = update_topk_correlated(m0, np.ones(3), member=0, k=2, beta2=1.0)
    np.testing.assert_allclose(math.exp(logZ), 0.04144731766, atol=1e-9)
    np.testing.assert_allclose(v[1:], 1.28354251492, atol=1e-6)


def test_winner_event_never_raises_a_variance():
    m0 = np.array([-4.0, 0.0, 0.0])
    _, v, _ = update_topk_correlated(m0, np.ones(3), member=0, k=1, beta2=1.0)
    _, vw, _ = update_winner_correlated(m0, np.ones(3), 0, np.zeros((3, 1)), beta2=1.0)
    np.testing.assert_allclose(v, vw, atol=1e-9)
    assert np.all(v <= 1.0 + 1e-12)
