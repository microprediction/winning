"""Ratings-filter regressions, round 3: shape contracts, contrast-space
evidence, unit equivariance, ordered-statistics tails and history
bookkeeping."""

import numpy as np
import pytest

from winning.ratings.full import update_order_full
from winning.ratings.nway import update_order_correlated
from winning.ratings.teams import update_team_order_full


# --- #456: singleton orders validate shapes before the tautology return

M3 = np.array([0.0, 1.0, 2.0])
S3 = np.eye(3)


@pytest.mark.parametrize("call", [
    lambda: update_order_full(M3, np.array([1.0, 2.0]), [1]),
    lambda: update_order_full(M3, S3, [1], V=np.ones((2, 2))),
    lambda: update_order_full(M3, S3, [1], beta2=np.ones(2)),
    lambda: update_order_correlated(M3, np.ones(3), [1], np.ones((2, 2))),
    lambda: update_team_order_full(M3, S3, np.ones((2, 2)), [1]),
    lambda: update_team_order_full(M3, S3, np.ones((2, 3)), [1],
                                   V=np.ones((3, 3))),
])
def test_singleton_order_refuses_malformed_shapes(call):
    with pytest.raises(ValueError):
        call()


def test_singleton_order_is_still_an_exact_identity():
    m, S, lz = update_order_full(M3, S3, [1], V=np.ones((3, 1)))
    assert lz == 0.0 and np.array_equal(m, M3) and np.array_equal(S, S3)
    m, v, lz = update_order_correlated(M3, np.ones(3), [2], np.ones((3, 1)))
    assert lz == 0.0 and np.array_equal(m, M3)
    m, S, lz = update_team_order_full(M3, S3, np.ones((2, 3)), [0])
    assert lz == 0.0 and np.array_equal(S, S3)
