"""The exact multinomial-probit boundaries refuse non-finite inputs (#901).

choice_loglik_and_score returned a NaN log-likelihood with NaN mu-scores
and exactly zero loading scores for a NaN or infinite utility, and
MNProbit accepted an infinite covariate, whose poisoned contrast Gram
matrix then produced the #581 "not identified" message for a column that
was simply unusable. Julia's GHK engine already refused these; now the
exact engines and the fitters do, in python, Julia and R.
"""
import numpy as np
import pytest

from winning.likelihood import choice_loglik_and_score
from winning.mnprobit import MNProbit

NONFINITE = [np.nan, np.inf, -np.inf]
X_OK = np.array([[[0.5], [0.0]], [[0.0], [1.0]], [[1.0], [0.0]], [[0.0], [1.0]]])
CHOICE = [0, 1, 0, 1]


@pytest.mark.parametrize("x", NONFINITE)
@pytest.mark.parametrize("J", [2, 3])
def test_non_finite_utility_is_refused_by_name_and_index(x, J):
    mu = np.zeros((2, J)); mu[1, J - 1] = x
    with pytest.raises(ValueError, match=r"mu\[1, %d\]" % (J - 1)):
        choice_loglik_and_score(mu, np.zeros((J, 1)), [0, 0])


@pytest.mark.parametrize("x", NONFINITE)
def test_non_finite_loading_or_variance_is_refused(x):
    mu = np.zeros((2, 3))
    with pytest.raises(ValueError):
        choice_loglik_and_score(mu, np.array([[x], [0.0], [0.0]]), [0, 0])
    with pytest.raises(ValueError):
        choice_loglik_and_score(mu, np.zeros((3, 1)), [0, 0], D=[1.0, x, 1.0])


@pytest.mark.parametrize("x", NONFINITE)
def test_constructor_refuses_a_non_finite_covariate_before_the_rank_check(x):
    X = X_OK.copy(); X[0, 0, 0] = x
    with pytest.raises(ValueError, match=r"X\[0, 0, 0\]") as info:
        MNProbit(X, CHOICE, intercepts=False, r=0)
    assert "identified" not in str(info.value)                   # not the #581 message


@pytest.mark.parametrize("x", NONFINITE)
def test_predict_refuses_a_non_finite_new_design(x):
    m = MNProbit(X_OK, CHOICE, intercepts=False, r=0)
    m.fit(maxiter=5)
    Xn = X_OK.copy(); Xn[1, 1, 0] = x
    with pytest.raises(ValueError, match=r"X\[1, 1, 0\]"):
        m.predict_proba(Xn)
    assert np.isfinite(m.predict_proba(X_OK)).all()             # the finite control


def test_finite_controls_are_unchanged():
    mu = np.array([[0.2, -0.1, 0.4], [1.0, 0.0, -1.0]])
    ll, dmu, dV = choice_loglik_and_score(mu, np.array([[0.3], [-0.2], [0.1]]), [2, 0])
    assert np.isfinite(ll) and np.isfinite(dmu).all() and np.isfinite(dV).all()
    m = MNProbit(X_OK, CHOICE, intercepts=False, r=0)
    assert m.fit(maxiter=5) is not None
