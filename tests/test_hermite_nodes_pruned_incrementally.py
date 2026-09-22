"""hermite_nodes builds the product rule one dimension at a time and
prunes as it goes (#155): the kept nodes, their order and their weights
are exactly those of the full tensor pruned once, and the tensor itself
is never materialised (Q=41 at rank 5 is 116M nodes and ~9 GiB before
pruning; it now takes 0.2 s and 1.06M nodes)."""
import numpy as np
import pytest

from winning.factor.core import hermite_nodes


def _tensor_then_prune(k, Q, prune=1e-7):
    x, w = np.polynomial.hermite_e.hermegauss(Q)
    w = w / np.sqrt(2.0 * np.pi)
    grids = np.meshgrid(*([x] * k), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for d in range(k):
        W *= w[np.searchsorted(x, F[:, d])]
    keep = W > prune * W.max()
    return F[keep], W[keep] / W[keep].sum()


@pytest.mark.parametrize("k,Q", [(2, 15), (2, 41), (3, 15), (3, 21), (4, 9), (4, 15)])
def test_incremental_pruning_equals_the_tensor_pruned_once(k, Q):
    F1, W1 = hermite_nodes(k, Q=Q)
    F0, W0 = _tensor_then_prune(k, Q)
    assert F1.shape == F0.shape
    assert np.array_equal(F1, F0)
    assert np.abs(W1 - W0).max() < 1e-15


def test_rank_five_order_41_fits_in_memory():
    F, W = hermite_nodes(5, Q=41)
    assert F.shape == (len(W), 5)
    assert len(W) < 2_000_000
    assert abs(W.sum() - 1.0) < 1e-12
    assert np.abs(W @ F).max() < 1e-12                # symmetric rule
    assert np.abs((W[:, None] * F * F).sum(axis=0) - 1.0).max() < 1e-4   # 3e-5: the 1e-7 pruning
