"""abilities_from_race does not converge when TWO runners hold nearly all
the mass, at any field size; three or more substantive runners converge.

Found 2026-09-21 migrating a client off the research engine: a 3-player
field with one missing price came back with the two favourites' win
probabilities SWAPPED (0.37/0.62 for targets 0.60/0.40) under the
skew-normal base, and the call only warned. First read as an n=3
defect; it is not. Normal base, points=500, D = 3.2^2, k dominant
runners spread 0.6..0.4 and the rest at eps:

    n=3,  k=2, eps=6e-4          max|q-p| 5.8e-2
    n=10, k=2, eps=1e-3 / 1e-4            1.7e-3 / 7.5e-3
    n=30, k=2, eps=1e-4                   1.1e-2
    n=10 or 30, k=3 or 4, eps=1e-4        converges, 15-28 iterations, ~2e-9
    n=10, k=2, rest at 0.20 / 0.05 / 0.02 converges in 12 / 23 / 41 iterations
    n_iter=500 at n=3                     1.6e-1, worse than 60 (oscillation)
    D=1 instead of 3.2^2                  same failure (scale-independent)

n = 2 has a closed form (#83); an EFFECTIVE pair does not reach it, goes
to the lattice, and the coordinate-Newton oscillates: with two runners
holding the mass, an update to one moves the other's probability nearly
one-for-one. A third substantive runner dilutes the coupling. The
boundary is smooth (iterations climb as the rest shrink), so this is a
late-contest field with two contenders left, not a toy. Likely fix in
the solver: couple the update for the dominant pair (a 2x2 Newton block)
or damp when the effective field is two.

Also observed, not pinned: `converged` is False at residual 2.3e-8
(eps=0.1, n=3) because the tolerance is on the log residual at 1e-8.
"""
import warnings

import numpy as np
import pytest

import winning.factor as wf


def _solve(p, D, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, info = wf.abilities_from_race(p, D=D, points=500, return_info=True, **kw)
        q = np.asarray(wf.race_probabilities(mu, D=D, points=500))
    return np.asarray(mu), info, float(np.abs(q - p).max())


def _field(n, k, eps):
    """k substantive runners spread 0.6..0.4, the other n-k at eps."""
    p = np.r_[np.linspace(0.6, 0.4, k), np.full(n - k, eps)]
    return p / p.sum(), np.full(n, 3.2 ** 2)


XFAIL = pytest.mark.xfail(strict=True, reason="coordinate-Newton oscillates "
                          "when two runners hold nearly all the mass")


@pytest.mark.parametrize("n,eps", [(3, 1e-2), (3, 1e-3), (3, 6e-4),
                                   (10, 1e-3), (10, 1e-4), (30, 1e-4)])
@XFAIL
def test_two_dominant_runners_reproduce_the_target(n, eps):
    p, D = _field(n, 2, eps)
    _, _, resid = _solve(p, D)
    assert resid < 1e-6


@pytest.mark.parametrize("n", [10, 30])
@pytest.mark.parametrize("k", [3, 4])
def test_three_or_more_substantive_runners_converge(n, k):
    """It is the number of runners carrying mass, not the field size and
    not the small probabilities: the same negligible tail converges."""
    p, D = _field(n, k, 1e-4)
    _, info, resid = _solve(p, D)
    assert info["converged"]
    assert resid < 1e-6


def test_the_boundary_is_smooth_not_a_cliff():
    """k=2 still converges while the rest carry real mass, and the
    iteration count climbs as they shrink -- a degrading regime that ends
    in the xfails above, which is why it reaches real late-contest fields."""
    iters = []
    for rest in (0.20, 0.05, 0.02):
        p = np.r_[[0.6, 0.4], np.full(8, rest)]
        p /= p.sum()
        _, info, resid = _solve(p, np.full(10, 3.2 ** 2))
        assert resid < 1e-6
        iters.append(info["iterations"])
    assert iters[0] < iters[1] < iters[2]


def test_more_iterations_do_not_help():
    """Oscillation, not slow convergence: the fix is the update, not n_iter."""
    p, D = _field(3, 2, 6e-4)
    _, _, r60 = _solve(p, D)
    _, _, r500 = _solve(p, D, n_iter=500)
    assert r60 > 1e-2, "the effective pair now converges at 60 iterations: lift the xfails"
    assert r500 >= 0.5 * r60


def test_failure_is_scale_independent():
    p, _ = _field(3, 2, 6e-4)
    _, _, r = _solve(p, np.ones(3))
    assert r > 1e-2, "the effective pair now converges at unit variance: lift the xfails"
