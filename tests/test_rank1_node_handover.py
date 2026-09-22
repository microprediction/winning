"""The rank-1 node rule hands over from Gauss-Hermite to the midpoint-
quantile grid at Q > 80 (sharpness ~10), not Q > 201 (sharpness ~25).

Found 2026-09-22 while prototyping importance sampling in factor space:
winning's default rule scored 3.9e-3 at sharpness 21 and 3.9e-4 at
sharpness 47 -- the SHARPER field ten times more accurate, because only
the sharper one had escalated. Swept on an 8-runner rank-1 field against
3M-path truth (se ~3e-4):

    sharp   GH at Q=8*sharp   midpoint grid at the same Q
     6.7     7.3e-4 (Q=54)          --
    10.5     8.3e-4 (Q=85)          --
    14.9     2.8e-3 (Q=120)      3.4e-4
    21.1     4.0e-3 (Q=169)      3.3e-4
    27.2       (escalated)       3.5e-4

GH is erratic past Q ~ 100, not slowly worsening (Q=201 gives 4e-5 at
sharp 14.9 and 2.4e-3 at 21.1). The grid is at the floor everywhere.
"""
import warnings

import numpy as np
import pytest

import winning.factor as wf

N = 8
RNG = np.random.default_rng(3)
MU = np.sort(RNG.normal(size=N)) * 0.5
V = RNG.normal(size=(N, 1))
VC = V[:, 0] - V[:, 0].mean()


def _truth(D, paths=1_500_000):
    r = np.random.default_rng(0)
    wins = np.zeros(N)
    done = 0
    while done < paths:
        m = min(500_000, paths - done)
        X = MU + r.normal(size=(m, 1)) @ V.T + np.sqrt(D) * r.normal(size=(m, N))
        wins += np.bincount(X.argmin(1), minlength=N)
        done += m
    return wins / paths


def _sharp(D):
    return float(np.sqrt(2) * np.max(np.abs(VC) / np.sqrt(D)))


@pytest.mark.parametrize("D_level", [0.1, 0.05, 0.03])       # sharpness ~15, 21, 27
def test_default_rule_is_at_the_truth_floor_in_the_former_gap(D_level):
    D = np.full(N, D_level)
    assert 10 < _sharp(D) < 30
    truth = _truth(D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(wf.race_probabilities(MU, V=V, D=D, points=257))
    assert np.abs(p - truth).max() < 1.2e-3      # 1.5M-path se ~4e-4; GH was 2.8-4.0e-3 here


def test_gauss_hermite_still_serves_the_smooth_regime():
    """Below the handover the GH rule is kept: at sharpness ~7 it is
    7.3e-4 against truth, as good as the alternatives."""
    D = np.full(N, 0.5)
    assert _sharp(D) < 10
    truth = _truth(D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(wf.race_probabilities(MU, V=V, D=D, points=257))
    assert np.abs(p - truth).max() < 1.5e-3
