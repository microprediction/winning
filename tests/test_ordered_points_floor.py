"""ordered_probabilities on a negative-correlation block: resolution.

Reported in #66. A regular-simplex block of k front-runners (Cov = -rho,
rank k-1) priced at k=3 prefixes succeeded at points=501 and raised
FloatingPointError below it, at every block size k = 2..5: the
trifecta's inner integral is a cumulative trapezoid, O(dx^2), and the
win race's 8-points-per-sd floor left 2-5e-3 of mass against a 1e-3
tolerance (the defect fell 4x per doubling).

The kernel now Richardson-extrapolates that term away. This file used to
pin the old floor (test_coarse_request_raises_today, written to fail
once the resolution rule was fixed, with the instruction to delete it,
and test_requests_below_need_are_promoted_to_need, which asserted the
coarse request still raised). They are replaced by what the issue's
review asked for instead: convergence against a fine reference, and
invariance under rotations of the loadings that leave the covariance
unchanged -- not exact `need` values.
"""
import numpy as np
import pytest

import winning.factor as wf

N, RHO = 9, 0.15
MU = np.linspace(-0.5, 0.5, N)


def negative_block(k):
    """k front-runners who cannot all lead: Cov(i, j) = -rho, as a regular
    simplex in rank k-1 (the construction from the issue)."""
    A = np.eye(k) - np.ones((k, k)) / k
    w, U = np.linalg.eigh(A)
    U = U[:, w > 1e-9]
    V = np.zeros((N, k - 1))
    V[np.arange(k)] = np.sqrt(RHO * k) * U
    D = np.ones(N)
    D[np.arange(k)] = 1.0 - RHO * (k - 1)
    return V, D


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_negative_block_prices_at_501(k):
    V, D = negative_block(k)
    T = wf.ordered_probabilities(MU, 3, V=V, D=D, points=501)
    assert T.shape == (N, N, N)
    assert np.all(T >= 0)
    assert abs(T.sum() - 1.0) < 1e-3
    for i in range(N):                       # repeated indices are impossible
        assert T[i, i, :].max() == 0 and T[i, :, i].max() == 0 and T[:, i, i].max() == 0


def _rotate(V, seed):
    """V Q for a random orthogonal Q: the same covariance, a different
    loading basis (the eigensolver's arbitrary choice)."""
    r = V.shape[1]
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(r, r)))
    return V @ Q


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_coarse_requests_price_and_converge(k):
    """points=65 and 129 used to raise "total mass ... defect 2-5e-3".
    They now price, and agree with a 2001-point reference cell by cell."""
    V, D = negative_block(k)
    ref = wf.ordered_probabilities(MU, 3, V=V, D=D, points=2001)
    for pts in (65, 129):
        T = wf.ordered_probabilities(MU, 3, V=V, D=D, points=pts)
        assert np.all(T >= 0)
        assert np.max(np.abs(T - ref)) < 1e-6, (k, pts)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_loading_rotation_invariance(k):
    """The internal `need` depends on the loading basis (it varied
    215-321 across rotations); the answer must not."""
    V, D = negative_block(k)
    base = wf.ordered_probabilities(MU, 3, V=V, D=D, points=129)
    for seed in (1, 2, 3):
        T = wf.ordered_probabilities(MU, 3, V=_rotate(V, seed), D=D,
                                     points=129)
        assert np.max(np.abs(T - base)) < 1e-5, (k, seed)


def test_extrapolated_kernel_matches_pure_path():
    """Richardson sits above the backend choice: compiled and NumPy
    kernels give the same extrapolated answer."""
    import winning
    V, D = negative_block(3)
    try:
        winning.use_rust(True)
        a = wf.ordered_probabilities(MU, 3, V=V, D=D, points=129)
        winning.use_rust(False)
        b = wf.ordered_probabilities(MU, 3, V=V, D=D, points=129)
    finally:
        winning.use_rust(True)
    assert np.max(np.abs(a - b)) < 1e-10
