"""New data must carry the design the model was FITTED on.

`predict_proba` decided whether to prepend generated alternative-intercept
columns from the COLUMN COUNT alone, and the model recorded neither
whether it had generated them nor how many covariates the caller
supplied. New data with the wrong number of features were therefore
silently REINTERPRETED rather than refused.

The guess is wrong exactly when the wrong width plus the `J - 1`
generated columns happens to equal the fitted width. Three covariates
fitted without intercepts, handed ONE, gained two synthetic intercept
columns and returned a plausible probability row — applying coefficients
fitted to the covariates to the intercepts and ignoring the supplied
feature entirely (#261).

Julia had the identical defect (#195) and was fixed first; this is the
sweep I should have done at the same time and did not.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.mnprobit import MNProbit


def _fitted(p_raw=3, J=3, intercepts=False):
    m = MNProbit(np.zeros((2, J, p_raw)), [0, 1],
                 intercepts=intercepts, r=1)
    m.params_ = np.linspace(0.7, -0.4, m.p)
    m.V_ = np.zeros((J, 1))
    return m


def test_the_model_records_what_it_was_built_with():
    m = _fitted()
    assert m.intercepts is False
    assert m.p_raw == 3
    assert m.p == 3
    mi = _fitted(p_raw=2, intercepts=True)
    assert mi.intercepts is True
    assert mi.p_raw == 2
    assert mi.p == 2 + (3 - 1)


def test_the_colliding_width_is_refused():
    """1 + (J - 1) == 3 == p is exactly the case the guess got wrong."""
    m = _fitted()
    with pytest.raises(ValueError, match="1 covariate columns"):
        m.predict_proba(np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1))


@pytest.mark.parametrize("nc", [1, 2, 4, 7])
def test_any_wrong_covariate_count_is_refused(nc):
    m = _fitted()
    with pytest.raises(ValueError, match="covariate columns"):
        m.predict_proba(np.zeros((1, 3, nc)))


def test_the_wrong_alternative_count_is_refused():
    m = _fitted()
    with pytest.raises(ValueError, match="alternatives"):
        m.predict_proba(np.zeros((1, 2, 3)))


def test_the_fitted_design_still_predicts():
    m = _fitted()
    P = m.predict_proba(np.arange(9.0).reshape(1, 3, 3))
    assert P.shape == (1, 3)
    assert (P >= 0).all()
    assert abs(P.sum() - 1) < 1e-12


def test_default_prediction_on_the_training_design_is_untouched():
    m = _fitted()
    P = m.predict_proba()
    assert P.shape == (2, 3)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-12


@pytest.mark.parametrize("spelling", ["raw", "full"])
def test_an_intercept_model_takes_either_spelling(spelling):
    mi = _fitted(p_raw=2, intercepts=True)
    nc = mi.p_raw if spelling == "raw" else mi.p
    P = mi.predict_proba(np.zeros((4, 3, nc)))
    assert P.shape == (4, 3)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-12


def test_an_intercept_model_refuses_anything_between():
    mi = _fitted(p_raw=2, intercepts=True)
    with pytest.raises(ValueError, match="covariate columns"):
        mi.predict_proba(np.zeros((4, 3, 3)))
