"""Finite factor weights whose raw SUM overflows are still one law (#415).

W and c*W describe the same factor law, but the normaliser divided by
W.sum(): for W = [1e308, 1e308] that total is inf, W / inf is all zero,
and every probability came back NaN. Normalising through max(W) first
keeps the relative-weight contract for every finite input.
"""
import numpy as np
import pytest

from winning.shapes import as_weights
from winning.factor import race_probabilities, abilities_from_race
from winning.factor.permutations import plackett_luce_prefix_logprob

MU = np.array([0.0, 0.4, 1.0])
V = np.array([[0.7], [0.0], [-0.4]])
D = np.ones(3)
F = np.array([[-1.0], [1.0]])
BIG = [1e308, 1e308]


def test_as_weights_survives_an_overflowing_total():
    assert np.array_equal(as_weights(BIG), [0.5, 0.5])
    assert np.allclose(as_weights([1e308, 1e308, 1e307]),
                       np.array([10, 10, 1]) / 21)
    with pytest.raises(ValueError, match="positive total"):
        as_weights([0.0, 0.0])


def test_forward_is_invariant():
    p1 = race_probabilities(MU, V=V, D=D, F=F, W=[1.0, 1.0], points=257)
    pb = race_probabilities(MU, V=V, D=D, F=F, W=BIG, points=257)
    assert np.all(np.isfinite(pb))
    assert np.max(np.abs(pb - p1)) < 1e-15
    assert np.allclose(p1, [0.51529828, 0.30546137, 0.17924036], atol=1e-7)


def test_inverse_is_invariant():
    p = np.array([0.5, 0.3, 0.2])
    a1 = abilities_from_race(p, V=V, D=D, F=F, W=[1.0, 1.0])
    ab = abilities_from_race(p, V=V, D=D, F=F, W=BIG)
    assert np.max(np.abs(ab - a1)) < 1e-12


def test_plackett_luce_branch_is_invariant():
    a = plackett_luce_prefix_logprob(MU, [0, 1], V=V, F=F, W=[1.0, 1.0])
    b = plackett_luce_prefix_logprob(MU, [0, 1], V=V, F=F, W=BIG)
    assert np.isfinite(b) and abs(a - b) < 1e-12
