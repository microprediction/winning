"""#1076: the correlated updates' rank <= 2 factor rule follows the
loading sharpness. Below _FACTOR_LADDER_BAND the fixed 7^r Hermite tensor
is kept to the bit; past it the tensor is recentred on the factor
posterior at _RECENTRED_QF[r] nodes per factor; across it the two are
blended with a weight that depends on the loadings and variances alone."""
import numpy as np
import pytest

import winning.ratings.nway as nw
from winning.likelihood import _smooth_sharpness


def _field(r, sharp, seed):
    rng = np.random.default_rng(seed)
    n = 5
    m = rng.normal(scale=0.5, size=n)
    v = rng.uniform(0.2, 1.0, size=n)
    V = rng.normal(size=(n, r))
    V *= sharp / _smooth_sharpness(V, v + 1.0)[0]
    return m, v, V, [int(x) for x in rng.permutation(n)]


@pytest.mark.parametrize("r", [1, 2])
def test_below_the_band_the_rule_is_untouched(r):
    m, v, V, _ = _field(r, 0.7, 1)
    F, W = nw._factor_grid(r)
    logW = np.log(W)
    F2, logW2 = nw._sharp_factor_rule(m, v, V, 1.0, None, F, logW)
    assert F2 is F and logW2 is logW


@pytest.mark.parametrize("r, qref", [(1, 81), (2, 31)])
def test_sharp_order_update_matches_converged_reference(r, qref):
    # main's fixed 7^r tensor: log Z off by 0.06-0.12 on these fields
    for seed in (2, 3):
        m, v, V, order = _field(r, 2.5, seed)
        mp, vp, lz = nw.update_order_correlated(m, v, order, V, 1.0)
        mr, vr, lzr = nw.update_order_correlated(m, v, order, V, 1.0, Qf=qref)
        assert abs(lz - lzr) < 2e-3
        assert np.abs(mp - mr).max() < 2e-3
        assert np.abs(vp - vr).max() < 5e-3


@pytest.mark.parametrize("r", [1, 2])
def test_no_jump_across_the_band(r):
    m, v, V0, order = _field(r, 1.0, 4)
    ss = np.linspace(0.76, 1.04, 29)
    z = np.array([nw.update_order_correlated(m, v, order, s * V0, 1.0)[2]
                  for s in ss])
    d1 = np.abs(np.diff(z))
    d2 = np.abs(np.diff(z, 2))
    # a rule switch shows as one step far off the trend; the blend's
    # second differences stay at the curve's own scale
    assert d2.max() < 0.05 * d1.max()


def test_caller_at_the_recentred_order_keeps_their_tensor(monkeypatch):
    m, v, V, order = _field(1, 2.5, 5)
    a = nw.update_order_correlated(m, v, order, V, 1.0, Qf=11)
    monkeypatch.setattr(nw, "_RECENTRED_QF", {})        # the plain tensor
    b = nw.update_order_correlated(m, v, order, V, 1.0, Qf=11)
    assert a[2] == b[2] and np.array_equal(a[0], b[0])


@pytest.mark.parametrize("r", [1, 2])
def test_sharp_winner_rule_stays_coherent(r):
    # winner events keep one fixed rule for every outcome, so the K
    # winner probabilities sum to one and the mean updates average to
    # the prior mean (a recentred rule broke both at 1e-4)
    m, v, V, _ = _field(r, 2.5, 6)
    res = [nw.update_winner_correlated(m, v, w, V, 1.0) for w in range(len(m))]
    p = np.array([np.exp(z) for _, _, z in res])
    assert abs(p.sum() - 1.0) < 1e-12
    drift = sum(pk * (mp - m) for pk, (mp, _, _) in zip(p, res))
    assert np.abs(drift).max() < 1e-10


@pytest.mark.parametrize("r, qref, tol", [(1, 81, 4e-3), (2, 31, 6e-3)])
def test_sharp_winner_update_is_closer_to_the_reference(r, qref, tol):
    for seed in (2, 3):
        m, v, V, _ = _field(r, 2.5, seed)
        a = nw.update_winner_correlated(m, v, 0, V, 1.0)
        b = nw.update_winner_correlated(m, v, 0, V, 1.0, Qf=qref)
        assert abs(a[2] - b[2]) < tol and np.abs(a[0] - b[0]).max() < tol
