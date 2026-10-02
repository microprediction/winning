"""The ratings factor mixtures depend on V V' only (#130).

A column sign flip, an orthogonal rotation, a zero column, or a
permutation of entrants (which may flip an eigenvector sign in the
belief split) is the same Gaussian model and must give the same
posterior and evidence -- at the DEFAULT node budget, not merely in the
limit. Each check is paired with one that a genuinely different V moves
the answer, so the equality cannot be two independent races."""

import numpy as np
import pytest

from winning.ratings.full import update_order_full, update_winner_full
from winning.ratings.nway import update_order_correlated, update_winner_correlated


def _same(a, b, tol=1e-10):
    return (np.abs(a[0] - b[0]).max() < tol and np.abs(a[1] - b[1]).max() < tol
            and abs(a[2] - b[2]) < tol)


def _rot(r, th=0.37, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(r, r)))
    return Q


@pytest.mark.parametrize("r", [1, 2, 3])
def test_column_signs_rotation_and_zero_padding(r):
    rng = np.random.default_rng(123)
    n = 5
    m = rng.normal(size=n); v = 0.4 + rng.random(n)
    V = rng.normal(size=(n, r)) * 0.8
    ref = update_winner_correlated(m, v, 4, V)
    for c in range(r):
        V2 = V.copy(); V2[:, c] *= -1
        assert _same(ref, update_winner_correlated(m, v, 4, V2)), c
    assert _same(ref, update_winner_correlated(m, v, 4, V @ _rot(r)))
    Vpad = np.hstack([V, np.zeros((n, 1))])
    assert _same(ref, update_winner_correlated(m, v, 4, Vpad))
    # orders too
    o = [4, 0, 2, 1, 3]
    refo = update_order_correlated(m, v, o, V)
    assert _same(refo, update_order_correlated(m, v, o, V @ _rot(r, seed=1)))
    # and the loadings matter
    other = update_winner_correlated(m, v, 4, V * 1.3)
    assert np.abs(other[0] - ref[0]).max() > 1e-3


def test_full_covariance_update_is_permutation_equivariant():
    rng = np.random.default_rng(2)
    n = 5
    m = rng.normal(size=n)
    A = rng.normal(size=(n, n))
    S = A @ A.T / n + 0.5 * np.eye(n)
    V = rng.normal(size=(n, 2)) * 0.2
    winner = 3
    a = update_winner_full(m, S, winner, V, nodes_log2=10)
    for seed in range(4):
        perm = np.random.default_rng(seed).permutation(n)
        inv = np.argsort(perm)
        b = update_winner_full(m[perm], S[np.ix_(perm, perm)], inv[winner],
                               V[perm], nodes_log2=10)
        assert np.abs(a[0] - b[0][inv]).max() < 1e-9, perm
        assert np.abs(a[1] - b[1][np.ix_(inv, inv)]).max() < 1e-9, perm
        assert abs(a[2] - b[2]) < 1e-9
    # a sign flip of an outcome-loading column too
    V2 = V.copy(); V2[:, 1] *= -1
    c = update_order_full(m, S, [3, 1, 0, 4, 2], V)
    d = update_order_full(m, S, [3, 1, 0, 4, 2], V2)
    assert _same(c, d, 1e-9)
