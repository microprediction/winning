"""MNP factor rank must be identified (#201).

With unit idiosyncratic variances and a zero reference row, the
strictly-lower-triangular loading block has r*J - r*(r+1)/2 free
entries while the differenced covariance has J*(J-1)/2 - 1 shape
degrees of freedom. At r >= J - 1 the loading block is full rank and
D = 1 no longer fixes the utility scale: (2 beta, V') prices every
choice exactly like (beta, V). The old default r = 2 was therefore
nonidentified for ordinary three-alternative data.
"""
import numpy as np
import pytest

from winning.mnprobit import MNProbit, MNProbitClassifier, max_identified_rank

A = np.array([[-1., 1., 0.], [-1., 0., 1.]])
V = np.array([[0., 0.], [.5, 0.], [.2, .7]])
Vp = np.array([[0., 0.], [2.6457513110645907, 0.],
               [1.2850792082313727, 2.543338638202044]])


def test_the_ridge_is_exact():
    c = A @ (V @ V.T + np.eye(3)) @ A.T
    cp = A @ (Vp @ Vp.T + np.eye(3)) @ A.T
    assert np.allclose(cp, 4 * c)


def test_max_identified_rank():
    assert [max_identified_rank(J) for J in (2, 3, 4, 5)] == [0, 1, 2, 3]


@pytest.mark.parametrize("J", [2, 3, 4, 5])
def test_admissible_and_refused_ranks(J):
    X = np.random.default_rng(J).normal(size=(6, J, 1))
    ch = np.arange(6) % J
    assert MNProbit(X, ch).r == min(2, J - 2)
    for r in range(J - 1):
        assert MNProbit(X, ch, r=r).r == r
    for r in (J - 1, J):
        with pytest.raises(ValueError, match="not identified"):
            MNProbit(X, ch, r=r)


def test_default_three_class_fit_is_identified():
    rng = np.random.default_rng(2)
    T = 400
    X = rng.normal(size=(T, 3, 1))
    U = 0.9 * X[:, :, 0] + rng.normal(size=(T, 3))
    m = MNProbit(X, U.argmax(axis=1), intercepts=False).fit(polish=False)
    assert m.r == 1
    assert abs(m.params_[0] - 0.9) < 0.3


def test_binary_fits_at_rank_zero():
    rng = np.random.default_rng(4)
    T = 300
    X = rng.normal(size=(T, 2, 1))
    U = 0.8 * X[:, :, 0] + rng.normal(size=(T, 2))
    m = MNProbit(X, U.argmax(axis=1), intercepts=False).fit(polish=False)
    assert m.r == 0
    assert abs(m.params_[0] - 0.8) < 0.25


def test_classifier_default_rank():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 3, 1))
    clf = MNProbitClassifier().fit(X, rng.integers(0, 3, 60))
    assert clf.model_.r == 1
    with pytest.raises(ValueError):
        MNProbitClassifier(r=2).fit(X, rng.integers(0, 3, 60))
