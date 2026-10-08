"""MNProbit estimator boundary: what fit() may be asked, and when it
may say it converged.

Each case was an ordinary-looking result for a request that had no
answer: a floored rank (#479), a NaN iteration cap read as success
(#560), an empty dataset (#591), a parameter-free null that crashed
(#495), a never-chosen alternative whose intercept has no finite MLE
(#496), mean-design columns the contrasts do not identify (#581), and a
classifier score that indexed past malformed labels (#492).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from winning.mnprobit import MNProbit, MNProbitClassifier, _check_rank


@pytest.mark.parametrize("bad", [1.1, 1.9, np.nan, np.inf, True, "1"])
def test_fractional_or_nonfinite_rank_is_refused(bad):
    with pytest.raises(ValueError):
        _check_rank(bad, 4)
    with pytest.raises(ValueError):
        MNProbit(np.zeros((2, 4, 1)), [0, 1], r=bad)


def test_integral_rank_spellings_are_accepted():
    assert _check_rank(1.0, 4) == 1
    assert _check_rank(np.int64(2), 4) == 2
    assert MNProbit(np.zeros((2, 4, 1)), [0, 1], r=2.0).r == 2


def _overlap():
    d = np.array([-2., -1., -.5, .5, 1., 2.])
    choice = np.array([0, 0, 1, 0, 1, 1])
    return np.stack([-d / 2, d / 2], axis=1)[..., None], choice


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.5, 1.5, -1, True])
def test_maxiter_must_be_a_whole_count(bad):
    X, choice = _overlap()
    with pytest.raises(ValueError, match="maxiter"):
        MNProbit(X, choice, intercepts=False, r=0).fit(maxiter=bad)


def test_maxiter_zero_does_no_work_and_does_not_converge():
    X, choice = _overlap()
    m = MNProbit(X, choice, intercepts=False, r=0).fit(maxiter=0)
    assert m.theta_[0] == 0.0 and not m.converged_
    m = MNProbit(X, choice, intercepts=False, r=0).fit(maxiter=100.0)
    assert m.converged_ and abs(m.theta_[0] - 1.2020830652) < 1e-6


@pytest.mark.parametrize("intercepts", [True, False])
def test_zero_observations_are_refused(intercepts):
    with pytest.raises(ValueError, match="at least one observation"):
        MNProbit(np.empty((0, 3, 0)), np.array([], dtype=int),
                 intercepts=intercepts, r=0)


def test_parameter_free_null_model_fits_without_an_optimiser():
    X = np.empty((4, 2, 0))
    m = MNProbit(X, [0, 1, 0, 1], intercepts=False, r=0).fit()
    assert m.converged_ and m.theta_.shape == (0,)
    assert m.loglik_ == pytest.approx(-4 * np.log(2), abs=1e-12)
    assert np.allclose(m.predict_proba(), 0.5)


def test_never_chosen_alternative_is_not_a_converged_fit():
    T = 100
    with pytest.warns(RuntimeWarning, match="never chosen"):
        m = MNProbit(np.zeros((T, 3, 0)), np.arange(T) % 2,
                     intercepts=True, r=0).fit(maxiter=500)
    assert not m.converged_ and m.boundary_
    assert m.never_chosen_.tolist() == [2]
    # the reference alternative unchosen is the same separation
    with pytest.warns(RuntimeWarning, match=r"\[0\]"):
        MNProbit(np.zeros((T, 3, 0)), 1 + np.arange(T) % 2, r=0).fit()


def test_every_alternative_chosen_still_converges_quietly():
    T = 99
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m = MNProbit(np.zeros((T, 3, 0)), np.arange(T) % 3, r=0).fit()
    assert m.converged_ and not m.boundary_ and m.never_chosen_.size == 0


T_, J_ = 120, 3
CHOICE = np.repeat([0, 1, 2], [30, 50, 40])


def test_covariate_duplicating_a_generated_intercept_is_refused():
    X = np.zeros((T_, J_, 1))
    X[:, 1, 0] = 1.0
    with pytest.raises(ValueError, match=r"X\[:, :, 0\].*not identified"):
        MNProbit(X, CHOICE, r=0).fit()


def test_covariate_common_to_all_alternatives_is_refused():
    z = np.random.default_rng(1).normal(size=T_)
    X = np.repeat(z[:, None, None], J_, axis=1)
    with pytest.raises(ValueError, match="not identified"):
        MNProbit(X, CHOICE, intercepts=False, r=0).fit()


def test_duplicate_covariate_is_refused_and_named():
    x = np.random.default_rng(2).normal(size=(T_, J_, 1))
    X = np.concatenate([x, -3.0 * x], axis=2)
    with pytest.raises(ValueError, match=r"X\[:, :, 1\]"):
        MNProbit(X, CHOICE, r=0).fit()


def test_full_rank_design_fits():
    X = np.random.default_rng(3).normal(size=(T_, J_, 2))
    assert MNProbit(X, CHOICE, r=0).fit().converged_


def test_classifier_score_uses_the_fit_label_contract():
    X = np.zeros((5, 2, 0))
    y = np.array([0, 0, 0, 0, 1])
    clf = MNProbitClassifier(r=0, intercepts=True).fit(X, y)
    assert clf.score(X, y) == pytest.approx(
        (4 * np.log(0.8) + np.log(0.2)) / 5, abs=1e-6)
    for bad in (np.array([-1, 0, 0, 0, 1]), y[:1], y[:3],
                np.append(y, 0), np.array([0, 0, 0, 0, 2]),
                np.array([0., 0., 0., 0., 0.5])):
        with pytest.raises(ValueError, match="choice"):
            clf.score(X, bad)
