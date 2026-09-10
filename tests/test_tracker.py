"""Consistency tests for the marginal ratings tracker (winning.ratings.tracker).

Each test guards a bug found (or a property proven) during development:
  - the exact win-node ranking factor agrees with the brute-force Gibbs
    data-augmentation posterior (order_augmented is the reference, not the
    production path);
  - the scores update is unbiased under performance noise alone (think like a
    Bayesian: the observation noise is beta2, the belief variance is the prior);
  - overlapping fields with class structure: per-contest recentering of the
    posterior means erased between-group ability differences (the varying-field
    washout bug) -- the tracker must recover the gap through overlap races;
  - prediction is invariant to a common shift of the stored means (gauge).
"""
import numpy as np
import pytest

from winning.ratings.tracker import (AbilityTracker, AbilityState,
                                     order_augmented, walk_forward)
from winning.ratings.nway import update_ranking_exact


def test_exact_ranking_matches_gibbs_reference():
    rng = np.random.default_rng(11)
    m = np.array([0.5, 0.2, -0.1, -0.3, -0.3])
    v = np.ones(5)
    beta2 = 1.0
    order = [0, 1, 2, 3, 4]
    Ee, Ve = update_ranking_exact(m.copy(), v.copy(), order, beta2=beta2)
    Eg, Vg = order_augmented(m.copy(), v.copy(), order, beta2, rng,
                             n_aug=4000, burn=500)
    c = lambda x: np.asarray(x, float) - np.asarray(x, float).mean()
    assert np.max(np.abs(c(Ee) - c(Eg))) < 0.06
    assert np.max(np.abs(np.asarray(Ve) - Vg)) < 0.12


def test_scores_update_unbiased_under_performance_noise():
    rng = np.random.default_rng(3)
    ids = list("01234567")
    n = len(ids)
    m0 = np.array([1.2, 0.8, 0.4, 0.1, -0.1, -0.4, -0.8, -1.2])
    beta2, v0, TR = 1.0, 1.0, 4000
    trk = AbilityTracker(beta2=beta2, drift=0.0)
    acc = np.zeros(n)
    for _ in range(TR):
        perf = -m0 + rng.normal(0, np.sqrt(beta2), n)
        trk.state = {i: AbilityState(float(m0[k]), v0, 0.0)
                     for k, i in enumerate(ids)}
        trk.observe(ids, 0.0, scores=perf.tolist())
        mm = np.array([trk.state[i].mean for i in ids])
        acc += mm - mm.mean()
    bias = acc / TR - (m0 - m0.mean())
    assert np.max(np.abs(bias)) < 0.05


def test_overlapping_fields_recover_class_gap():
    # Two groups, gap 1.5, 10% cross-group races. Per-contest recentering
    # (the washout bug) gives est. gap ~0 and corr ~0.5 on this setup.
    rng = np.random.default_rng(3)
    NG, GAP, BETA = 25, 1.5, 1.0
    abil = np.concatenate([rng.normal(+GAP / 2, 0.6, NG),
                           rng.normal(-GAP / 2, 0.6, NG)])
    contests = []
    for k in range(900):
        u = rng.random()
        pool = (np.arange(0, NG) if u < 0.45 else
                np.arange(NG, 2 * NG) if u < 0.90 else np.arange(0, 2 * NG))
        ids = rng.choice(pool, 8, replace=False)
        perf = -abil[ids] + rng.normal(0, np.sqrt(BETA), 8)
        contests.append({"runners": [str(i) for i in ids], "t": float(k),
                         "scores": perf.tolist()})
    out = walk_forward(contests, warmup=200, beta2=BETA, drift=0.0,
                       market_arm=False)
    trk = out["tracker"]
    m = np.array([trk.state[str(i)].mean if str(i) in trk.state else np.nan
                  for i in range(2 * NG)])
    gap_est = np.nanmean(m[:NG]) - np.nanmean(m[NG:])
    ok = ~np.isnan(m)
    corr = np.corrcoef(m[ok] - m[ok].mean(), abil[ok] - abil[ok].mean())[0, 1]
    assert gap_est > 0.3, f"class gap washed out: {gap_est:.3f}"
    assert corr > 0.8, f"rating-truth corr too low: {corr:.3f}"


def test_predict_gauge_invariant():
    trk = AbilityTracker(beta2=1.0, drift=0.0)
    ids = list("abcde")
    for i, mm in zip(ids, [0.5, -0.3, 0.1, 0.9, -1.2]):
        trk.state[i] = AbilityState(mm, 1.0, 0.0)
    p1 = trk.predict(ids, 1.0)
    for i in ids:
        s = trk.state[i]
        trk.state[i] = AbilityState(s.mean + 7.0, s.var, s.last_time)
    p2 = trk.predict(ids, 1.0)
    assert np.max(np.abs(p1 - p2)) < 1e-10
    assert abs(p1.sum() - 1.0) < 1e-6
