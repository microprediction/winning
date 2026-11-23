"""Single-row Jacobians and the reduced-rank representation take an
integer runner index in [0, n) (#846). int(y) read 0.9 as runner 0 and
True as runner 1, -1 meant the last runner, and reduced_rank_representation
with i=-1 excluded nobody: an n-row rectangle and a 0.26 wrong winner
probability. Valid indices, Python or NumPy, are unchanged."""
import numpy as np
import pytest

from winning.alternatives import reduced_rank_representation
from winning.factor.polish import race_jacobian, race_jacobian_row
from winning.factor.topk import (top_k_jacobian, top_k_jacobian_row,
                                 top_k_jacobian_row_sigma)

MU = np.array([-0.7, 0.1, 0.9])
V = np.array([[0.4, -0.2], [-0.1, 0.3], [0.2, 0.5]])
D = np.array([0.8, 1.1, 0.6])
BAD = [-1, 3, 0.9, 1.9, 1.0, True, False, np.True_, np.nan, "1", None]


def _calls(i):
    return [lambda: race_jacobian_row(MU, i, points=129),
            lambda: top_k_jacobian_row(MU, i, 2, points=129),
            lambda: top_k_jacobian_row_sigma(MU, i, 2, points=129),
            lambda: reduced_rank_representation(MU, V, D, i)]


@pytest.mark.parametrize("i", BAD, ids=repr)
@pytest.mark.parametrize("which", range(4))
def test_invalid_runner_index_refused(i, which):
    with pytest.raises(ValueError, match="runner index"):
        _calls(i)[which]()


@pytest.mark.parametrize("i", [0, 1, 2, np.int64(2), np.int32(1)], ids=repr)
def test_valid_index_matches_full_jacobian_row(i):
    J = race_jacobian(MU, points=129)
    assert np.array_equal(race_jacobian_row(MU, i, points=129),
                          race_jacobian_row(MU, int(i), points=129))
    assert np.allclose(race_jacobian_row(MU, i, points=129), J[int(i)])
    Jk = top_k_jacobian(MU, 2, points=129)
    assert np.array_equal(top_k_jacobian_row(MU, i, 2, points=129),
                          Jk[int(i)])
    a, b = top_k_jacobian_row_sigma(MU, i, 2, points=129)
    a0, b0 = top_k_jacobian_row_sigma(MU, int(i), 2, points=129)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)
    r = reduced_rank_representation(MU, V, D, i)
    assert r["B"].shape == (2, 3) and r["upper"].shape == (2,)
    mask = np.arange(3) != int(i)
    assert np.array_equal(r["upper"], MU[int(i)] - MU[mask])
