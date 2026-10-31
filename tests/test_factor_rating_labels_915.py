"""Validate categorical labels before coercion at the public ratings boundaries."""
import numpy as np
import pytest

from winning.ratings import factor_ratings as fr

B = np.array([[2.0], [-1.0], [0.5]])
Z = np.eye(3)
BAD = [
    [0.9, 1, 2], [False, 1, 2], [np.bool_(False), 1, 2],
    ['0', '1', '2'], [0, 1, np.nan], [0, 1, np.inf],
    [0, 1, -np.inf], [0, 1, -1], [0, 1, 3], [0, 1, 1],
    [0, 1, 10**100], [0, 1, 2+0j], [[0, 1, 2]], 0, [],
]
SUBSET_CALLS = [
    lambda ids: fr.factor_design(ids, [1.0], 3, 1),
    lambda ids: fr.factor_abilities(B, ids, [1.0]),
    lambda ids: fr.predict_factor(B, ids, [1.0]),
    lambda ids: fr.factor_loglik(B, [(ids, [0], [1.0])]),
    lambda ids: fr.fit_factor_ratings([(ids, [0], [1.0])], 3, 1, n_iter=1),
    lambda ids: fr.factor_se(B, [(ids, [0], [1.0])]),
]
ORDER_CALLS = [
    lambda order: fr.fit_design_ratings([(Z, order)], 3, n_iter=1),
    lambda order: fr.design_loglik(B.ravel(), [(Z, order)]),
    lambda order: fr.design_se(B.ravel(), [(Z, order)], 3),
    lambda order: fr.factor_loglik(B, [([0, 1, 2], order, [1.0])]),
    lambda order: fr.fit_factor_ratings([([0, 1, 2], order, [1.0])], 3, 1, n_iter=1),
    lambda order: fr.factor_se(B, [([0, 1, 2], order, [1.0])]),
]


@pytest.mark.parametrize('call', SUBSET_CALLS)
@pytest.mark.parametrize('bad', BAD)
def test_invalid_entity_labels_rejected(call, bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize('call', ORDER_CALLS)
@pytest.mark.parametrize('bad', BAD)
def test_invalid_order_labels_rejected(call, bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint64, float, object])
@pytest.mark.parametrize('order', [[0], [2, 0], [2, 0, 1]])
def test_valid_labels_preserve_partial_orders_and_outputs(dtype, order):
    subset = np.array([2, 0, 1], dtype=dtype)
    ranking = np.array(order, dtype=dtype)
    subset_before, order_before = subset.copy(), ranking.copy()
    baseline = [([2, 0, 1], order, [1.0])]
    events = [(subset, ranking, [1.0])]
    np.testing.assert_array_equal(fr.factor_abilities(B, subset, [1.0]), B[[2, 0, 1], 0])
    np.testing.assert_array_equal(fr.factor_design(subset, [1.0], 3, 1).toarray(), Z[[2, 0, 1]])
    np.testing.assert_array_equal(fr.predict_factor(B, subset, [1.0]), fr.predict_factor(B, [2, 0, 1], [1.0]))
    np.testing.assert_array_equal(fr.factor_loglik(B, events), fr.factor_loglik(B, baseline))
    np.testing.assert_array_equal(fr.fit_factor_ratings(events, 3, 1, n_iter=2), fr.fit_factor_ratings(baseline, 3, 1, n_iter=2))
    np.testing.assert_array_equal(fr.design_loglik(B.ravel(), [(Z, ranking)]), fr.design_loglik(B.ravel(), [(Z, order)]))
    np.testing.assert_array_equal(fr.fit_design_ratings([(Z, ranking)], 3, n_iter=2), fr.fit_design_ratings([(Z, order)], 3, n_iter=2))
    np.testing.assert_array_equal(subset, subset_before)
    np.testing.assert_array_equal(ranking, order_before)


@pytest.mark.parametrize('call', SUBSET_CALLS + ORDER_CALLS)
@pytest.mark.parametrize('mask', [[False, True, False], True])
def test_masked_labels_are_refused_before_array_coercion(call, mask):
    labels = np.ma.array([0, 1, 2], mask=mask)
    before = labels.copy()
    with pytest.raises(ValueError, match="masked"):
        call(labels)
    np.testing.assert_array_equal(labels.data, before.data)
    np.testing.assert_array_equal(labels.mask, before.mask)


@pytest.mark.parametrize('call', SUBSET_CALLS + ORDER_CALLS)
@pytest.mark.parametrize('mask', [np.ma.nomask, [False, False, False]])
def test_unmasked_masked_arrays_preserve_valid_outputs(call, mask):
    labels = np.ma.array([0, 1, 2], mask=mask)
    actual, expected = call(labels), call([0, 1, 2])
    if hasattr(actual, 'toarray'):
        actual, expected = actual.toarray(), expected.toarray()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels.data, [0, 1, 2])
    assert not np.ma.getmaskarray(labels).any()
