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


# --- #648: an omitted runner cannot move a partial-order likelihood

@pytest.mark.parametrize("om,osd", [(0.0, 1.0), (0.0, 1e4), (1e6, 1.0),
                                    (-1e6, 1.0), (0.0, 1e6)])
def test_partial_order_ignores_omitted_runner(om, osd):
    from winning.ratings.nway import _order_pass_batch, order_loglik
    m = np.array([.4, .1, -.2])
    sd = np.array([1., .8, 1.2])
    ll0, g0 = order_loglik(m, sd, [0, 1, 2])
    assert abs(ll0 - (-1.2306349818290565)) < 1e-6
    for pos in range(4):
        mm, ss = np.insert(m, pos, om), np.insert(sd, pos, osd)
        o = [i if i < pos else i + 1 for i in range(3)]
        ll, g = order_loglik(mm, ss, o)
        lb, gb = _order_pass_batch(mm[None, :], ss, o)
        assert ll == pytest.approx(ll0, abs=1e-12) and ll <= 0.0
        assert lb[0] == pytest.approx(ll0, abs=1e-9)
        assert np.allclose(np.delete(g, pos), g0, atol=1e-10)
        assert g[pos] == 0.0 and gb[0, pos] == 0.0
