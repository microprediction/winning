"""abilities_from_race converges when two runners hold nearly all the mass.

It did not, at any field size. Found 2026-09-21 migrating a client off
the research engine: a 3-player field with one missing price came back
with the two favourites' win probabilities SWAPPED (0.37/0.62 for
targets 0.60/0.40) under the skew-normal base, and the call only warned.

The solver already damped the N = 2 case (alpha 0.7) because the K_2
Jacobi update has eigenvalue -1 and two-cycles -- but keyed on N == 2
literally. An EFFECTIVE pair at any N has the same eigenvalue in the
limit and got alpha = 1. Measured before the fix, normal base, 500
points, k dominant runners spread 0.6..0.4 and the rest at eps:

    n=3,  k=2, eps=6e-4          max|q-p| 5.8e-2
    n=10, k=2, eps=1e-3 / 1e-4            1.7e-3 / 7.5e-3
    n=30, k=2, eps=1e-4                   1.1e-2
    n_iter=500 at n=3                     1.6e-1, worse than 60 (oscillation)
    n=10 or 30, k=3 or 4, eps=1e-4        converged, 15-28 sweeps
    n=10, k=2, rest 0.20 / 0.05 / 0.02    converged in 12 / 23 / 41 sweeps

The damping is now keyed on the target's top-two share (> 0.8). Same
cases after: 19-20 sweeps to ~4e-9, and every field below the threshold
takes exactly the sweeps it took before. These tests pin both halves.
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


@pytest.mark.parametrize("n,eps", [(3, 1e-2), (3, 1e-3), (3, 6e-4), (3, 1e-4),
                                   (10, 1e-3), (10, 1e-4), (30, 1e-4)])
def test_two_dominant_runners_reproduce_the_target(n, eps):
    p, D = _field(n, 2, eps)
    _, info, resid = _solve(p, D)
    assert info["converged"], f"n={n} eps={eps}: {info}"
    assert resid < 1e-6
    assert info["iterations"] <= 30           # 19-20 measured; was 60 and failing


def test_the_favourites_come_out_in_the_right_order():
    """The client-visible symptom: A at 0.60 and B at 0.40 came back 0.37
    and 0.62 under the skew base. Min-wins, so the higher target needs the
    LOWER ability."""
    from winning.factor.races import skew_normal_base
    from scipy.stats import skewnorm
    a = 2.0
    p = np.array([1 / 2.0, 1 / 3.0, 1 / 2000.0]); p /= p.sum()
    D = np.full(3, 3.2 ** 2 * skewnorm(a).var())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.asarray(wf.abilities_from_race(p, D=D, base=skew_normal_base(a), points=500))
        q = np.asarray(wf.race_probabilities(mu, D=D, base=skew_normal_base(a), points=500))
    assert mu[0] < mu[1] < mu[2]
    assert np.abs(q - p).max() < 1e-6


def test_the_skew_case_that_started_this_round_trips():
    """Same target through the front door under the skew base, without the
    ordering hint: the race reproduces the prices."""
    from winning.factor.races import skew_normal_base
    from scipy.stats import skewnorm
    p = np.array([0.6, 0.4, 6e-4]); p /= p.sum()
    D = np.full(3, 3.2 ** 2 * skewnorm(2.0).var())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = wf.abilities_from_race(p, D=D, base=skew_normal_base(2.0), points=500)
        q = np.asarray(wf.race_probabilities(mu, D=D, base=skew_normal_base(2.0), points=500))
    assert np.abs(q - p).max() < 1e-6


@pytest.mark.parametrize("n", [10, 30])
@pytest.mark.parametrize("k", [3, 4])
def test_three_or_more_substantive_runners_still_converge(n, k):
    p, D = _field(n, k, 1e-4)
    _, info, resid = _solve(p, D)
    assert info["converged"]
    assert resid < 1e-6


def test_fields_below_the_threshold_are_untouched():
    """An ordinary field takes exactly the sweeps measured: 10 at n=8 (13
    before the sweeps learned to read their own contraction, #149) and 7
    at n=150 on these seeds (the function counts sweeps from 1). A change
    here means the gate or the damping rule moved; say which."""
    rng = np.random.default_rng(0)
    p = np.exp(rng.normal(size=8) * 0.8); p /= p.sum(); D = 0.5 + rng.random(8)
    _, info8, r8 = _solve(p, D)
    p = np.exp(rng.normal(size=150) * 1.0); p /= p.sum(); D = 0.6 + rng.random(150)
    _, info150, r150 = _solve(p, D)
    assert np.sort(p)[-2:].sum() < 0.8
    assert info8["converged"] and info150["converged"]
    assert info8["iterations"] == 10 and info150["iterations"] == 7
    assert max(r8, r150) < 1e-6


def test_the_degrading_regime_is_now_fast():
    """k=2 with the rest at 0.02 each (share 0.86) took 41 undamped sweeps
    and converged only just; damped it takes ~14."""
    p = np.r_[[0.6, 0.4], np.full(8, 0.02)]; p /= p.sum()
    _, info, resid = _solve(p, np.full(10, 3.2 ** 2))
    assert info["converged"] and resid < 1e-6
    assert info["iterations"] <= 20
