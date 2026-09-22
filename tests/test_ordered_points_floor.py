"""ordered_probabilities on a negative-correlation block: the points floor.

Reported in #66. A regular-simplex block of k front-runners (Cov = -rho,
rank k-1) priced at k=3 prefixes succeeds at points=501 and raises
FloatingPointError below it, at every block size k = 2..5.

Two facts pinned here, both measured on main at 2026-09-21:

1. A request below `need` is silently promoted to `need`, so the result
   is IDENTICAL at 65 and 257 points for k >= 4 -- lowering `points`
   there changes nothing. need = ceil(span / (sd_min / 8)) + 1 was
   calibrated for the win race, whose lattice quadrature is spectral;
   the 3-prefix kernel's inner integral is a cumsum, first-order, and
   8 points per sd leaves 2-5e-3 of mass on the table against a 1e-3
   tolerance.

2. A coarse request therefore raises. When the kernel's resolution rule
   is fixed, test_coarse_request_raises_today FAILS -- that is the
   signal to delete it and lower the documented floor.
"""
import numpy as np
import pytest

import winning.factor as wf
from winning.factor.races import _setup

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


def need_for(V, D):
    """The internal floor ordered_probabilities promotes `points` to."""
    mu, V, D, F, W, fn, left, right = _setup(MU, V, D, None, None, "normal")
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    span = float(M_all.max() - M_all.min()) + (left + right) * sd.max()
    return int(np.ceil(span / (float(sd.min()) / 8.0))) + 1


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_negative_block_prices_at_501(k):
    V, D = negative_block(k)
    T = wf.ordered_probabilities(MU, 3, V=V, D=D, points=501)
    assert T.shape == (N, N, N)
    assert np.all(T >= 0)
    assert abs(T.sum() - 1.0) < 1e-3
    for i in range(N):                       # repeated indices are impossible
        assert T[i, i, :].max() == 0 and T[i, :, i].max() == 0 and T[:, i, i].max() == 0


@pytest.mark.parametrize("k", [4, 5])
def test_requests_below_need_are_promoted_to_need(k):
    """Both requests run at `need`, so the results are bit-identical and
    the FloatingPointError message reports the same defect for both."""
    V, D = negative_block(k)
    need = need_for(V, D)
    assert need > 257, f"premise: need={need} must exceed both requests"
    msgs = []
    for pts in (65, 257):
        with pytest.raises(FloatingPointError) as e:
            wf.ordered_probabilities(MU, 3, V=V, D=D, points=pts)
        msgs.append(str(e.value))
    assert msgs[0] == msgs[1], "65 and 257 points gave different defects, so " \
                               "the request is no longer promoted to need"


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_coarse_request_raises_today(k):
    """Pins the documented floor. If this fails, the kernel resolves the
    3-prefix mass at fewer points than it used to: delete this test and
    lower the floor in the docs and in #66."""
    V, D = negative_block(k)
    with pytest.raises(FloatingPointError, match=r"total mass"):
        wf.ordered_probabilities(MU, 3, V=V, D=D, points=129)
