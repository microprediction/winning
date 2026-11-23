"""Entity and finishing-order labels are validated before any integer cast (#915).

np.asarray(labels, dtype=int) turned 0.9 into 0, 2.8 into 2, True into 1
and "2" into 2, so a fractional, Boolean or string label was silently
redirected to another entity or finisher, and every exported factor and
design ratings call returned finite, plausible output for a model the
caller never specified.
"""
import numpy as np
import pytest

from winning.ratings.factor_ratings import (design_loglik, factor_abilities, factor_design,
                                            factor_loglik, fit_design_ratings, fit_factor_ratings,
                                            predict_factor)

B = np.array([[2.0], [-1.0], [0.5]])
GOOD = [([0, 1, 2], [0, 2], [1.0])]
Z = np.eye(3)

BAD_LABELS = [[0.9, 1.1, 2.8], [True, False, 2], ["0", "1", "2"], [0, 1, float("nan")],
              [0, 1, float("inf")], [0, None, 2], [[0, 1, 2]]]


@pytest.mark.parametrize("subset", BAD_LABELS)
def test_malformed_entity_labels_are_refused_at_every_exported_door(subset):
    with pytest.raises(ValueError, match="subset"):
        factor_abilities(B, subset, [1.0])
    with pytest.raises(ValueError, match="subset"):
        predict_factor(B, subset, [1.0], points=65)
    with pytest.raises(ValueError, match="subset"):
        factor_design(subset, [1.0], 3, 1)
    with pytest.raises(ValueError, match="subset"):
        factor_loglik(B, [(subset, [0, 2], [1.0])])
    with pytest.raises(ValueError, match="subset"):
        fit_factor_ratings([(subset, [0, 2], [1.0])], 3, 1, n_iter=2)


@pytest.mark.parametrize("order", [[0.9, 2.1], [False, 2], ["0", "2"], [0, float("nan")], [[0, 2]]])
def test_malformed_orders_are_refused_in_the_factor_and_design_forms(order):
    with pytest.raises(ValueError, match="order"):
        factor_loglik(B, [([0, 1, 2], order, [1.0])])
    with pytest.raises(ValueError, match="order"):
        fit_factor_ratings([([0, 1, 2], order, [1.0])], 3, 1, n_iter=2)
    with pytest.raises(ValueError, match="order"):
        fit_design_ratings([(Z, order)], 3, n_iter=1)
    with pytest.raises(ValueError, match="order"):
        design_loglik(np.zeros(3), [(Z, order)])


def test_negative_and_out_of_range_labels_are_still_refused():
    with pytest.raises(ValueError):
        factor_abilities(B, [0, 1, 3], [1.0])
    with pytest.raises(ValueError):
        factor_abilities(B, [-1, 1, 2], [1.0])
    with pytest.raises(ValueError):
        factor_loglik(B, [([0, 1, 2], [0, 3], [1.0])])


def test_integral_floats_are_the_same_labels_as_ints():
    # a label column read from a frame arrives as float; 2.0 is the label 2
    as_float = [([0.0, 1.0, 2.0], [0.0, 2.0], [1.0])]
    assert factor_loglik(B, as_float) == factor_loglik(B, GOOD)
    np.testing.assert_array_equal(fit_factor_ratings(as_float, 3, 1, n_iter=2),
                                  fit_factor_ratings(GOOD, 3, 1, n_iter=2))
    np.testing.assert_array_equal(factor_design([0.0, 1.0, 2.0], [1.0], 3, 1).toarray(),
                                  factor_design([0, 1, 2], [1.0], 3, 1).toarray())
    np.testing.assert_array_equal(factor_abilities(B, np.array([2, 0], dtype=np.int32), [1.0]),
                                  factor_abilities(B, [2, 0], [1.0]))


def test_the_bad_labels_were_a_different_model():
    # the issue: [0.9, 1.1, 2.8] used to price exactly as [0, 1, 2]; a
    # legitimately different subset prices differently, so silence hid a
    # real change
    a = predict_factor(B, [0, 1, 2], [1.0], points=65)
    b = predict_factor(B, [1, 2, 0], [1.0], points=65)
    assert np.abs(a - b).max() > 1e-3
