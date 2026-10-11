"""loc_scale_from_topk_pair's D0 goes through as_idio like every other
variance: a scalar D0 failed with 'input arrays have different
dimensions' (#254)."""
import numpy as np
import pytest

from winning.factor.topk import loc_scale_from_topk_pair


def test_scalar_d0_is_the_broadcast_vector():
    q1, q2 = [0.5, 0.3, 0.2], [0.8, 0.7, 0.5]
    a = loc_scale_from_topk_pair(q1, 1, q2, 2, D0=1.0, n_iter=2,
                                 return_info=True)
    b = loc_scale_from_topk_pair(q1, 1, q2, 2, D0=np.ones(3), n_iter=2,
                                 return_info=True)
    for x, y in zip(a[:2], b[:2]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("bad", [0.0, np.nan, -1.0, [1.0, 1.0]])
def test_invalid_d0_is_refused(bad):
    with pytest.raises(ValueError):
        loc_scale_from_topk_pair([0.5, 0.3, 0.2], 1, [0.8, 0.7, 0.5], 2,
                                 D0=bad, n_iter=2)
