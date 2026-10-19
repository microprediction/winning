"""Dense-filter contracts, round 9: walk_forward from the prior
(warmup=0, #770), predict_race's clock (#772), tune_history refusing to
report an uncalibrated result (#774), and a resumed rate_history reading
its saved drift hyperparameters (#776). Each regression sits next to a
check that ordinary inputs nearby are unchanged."""

import math

import numpy as np
import pytest

import winning.ratings.history as H
from winning.ratings.history import (diffuse_full, predict_race,
                                     rate_history, tune_history, walk_forward)

RACES = [
    {"t": 0.0, "runners": ["a", "b", "c"], "scores": [1.0, 0.0, -0.5],
     "p_market": [0.5, 0.3, 0.2]},
    {"t": 2.0, "runners": ["a", "b", "c"], "order": [2, 0, 1]},
    {"t": 3.0, "runners": ["a", "c"], "winner": 1, "p_market": [0.6, 0.4]},
    {"t": 5.0, "runners": ["b", "c", "a"], "margins": [0.0, 1.5, 3.0]},
]


# ---- #772: predict_race clock ------------------------------------------

def _state():
    return rate_history(
        [{"t": 10.0, "runners": ["a", "b"], "scores": [1.0, 0.0]}],
        ids=["a", "b"], timescale=10.0, prior_var=1.0, beta2=1.0,
        lengths_scale=1.0, return_state=True)[2]


@pytest.mark.parametrize("t", [5.0, 9.999, np.nan, np.inf, -np.inf])
def test_predict_race_rejects_stale_or_non_finite_time(t):
    with pytest.raises(ValueError, match="prediction time"):
        predict_race(_state(), ["a", "b"], t=t)


def test_predict_race_in_order_times_unchanged():
    st = _state()
    p_none, _ = predict_race(st, ["a", "b"])
    p_eq, _ = predict_race(st, ["a", "b"], t=10.0)
    p_int, _ = predict_race(st, ["a", "b"], t=10)
    np.testing.assert_array_equal(p_none, p_eq)
    np.testing.assert_array_equal(p_none, p_int)
    # a later time diffuses exactly as diffuse_full says
    m, S = diffuse_full(st["m"], st["S"], dt=1.0, timescale=10.0)
    st2 = dict(st, m=m, S=S, t=11.0)
    np.testing.assert_array_equal(predict_race(st, ["a", "b"], t=11.0)[0],
                                  predict_race(st2, ["a", "b"])[0])


# ---- #774: tune_history with no valid evaluation -----------------------

def test_tune_history_raises_on_invalid_data():
    races = [{"t": 0.0, "runners": ["a", "b"],
              "scores": [float("nan"), 0.0]}]
    with pytest.raises(ValueError, match="not finite"):
        tune_history(races, tune=("tau2",), maxiter=30)


def test_tune_history_raises_on_unsupported_fixed_base():
    races = [{"t": 0.0, "runners": ["a", "b"], "winner": 0}]
    with pytest.raises(NotImplementedError):
        tune_history(races, tune=("tau2",), maxiter=30, base="laplace")


def test_tune_history_raises_on_invalid_fixed_scale():
    races = [{"t": 0.0, "runners": ["a", "b"], "scores": [1.0, 0.0]}]
    with pytest.raises(ValueError, match="lengths_scale"):
        tune_history(races, tune=("tau2",), maxiter=30, lengths_scale=-1)


def test_tune_history_still_penalizes_failing_trial_points(monkeypatch):
    # a valid baseline with some rejected trial points still calibrates,
    # identically to the unpatched run when the rejections are off-path
    want = tune_history(RACES, tune=("beta2",), maxiter=20)
    real = H.rate_history

    def flaky(races, **kw):
        if kw["beta2"] > 1e6:
            raise FloatingPointError("trial point")
        return real(races, **kw)

    monkeypatch.setattr(H, "rate_history", flaky)
    got = tune_history(RACES, tune=("beta2",), maxiter=20)
    assert got == want and np.isfinite(got[1])


# ---- #776: resumed drift hyperparameters -------------------------------

TWO = [{"t": 0.0, "runners": ["a", "b"], "scores": [1.0, 0.0]},
       {"t": 10.0, "runners": ["a", "b"], "scores": [0.0, 1.0]}]
KW = dict(ids=["a", "b"], prior_mean=3.0, prior_var=4.0, timescale=10.0,
          beta2=1.0, lengths_scale=1.0)


def test_resume_reads_saved_drift_hyperparameters():
    _, z, one = rate_history(TWO, return_state=True, **KW)
    _, z1, first = rate_history(TWO[:1], return_state=True, **KW)
    _, z2, two = rate_history(TWO[1:], state=first, return_state=True,
                              beta2=1.0, lengths_scale=1.0)
    np.testing.assert_array_equal(two["m"], one["m"])
    np.testing.assert_array_equal(two["S"], one["S"])
    assert z1 + z2 == pytest.approx(z, abs=1e-14)
    assert (two["prior_mean"], two["prior_var"], two["timescale"]) == \
        (3.0, 4.0, 10.0)


def test_resume_explicit_override_still_wins():
    _, _, first = rate_history(TWO[:1], return_state=True, **KW)
    kw = dict(KW, timescale=50.0); kw.pop("ids")
    _, _, a = rate_history(TWO[1:], state=first, return_state=True, **kw)
    _, _, b = rate_history(TWO[1:], state=dict(first, timescale=50.0),
                           return_state=True, beta2=1.0, lengths_scale=1.0,
                           prior_mean=3.0, prior_var=4.0)
    assert a["timescale"] == 50.0
    np.testing.assert_array_equal(a["m"], b["m"])


def test_fresh_run_defaults_unchanged():
    got = rate_history(RACES, return_state=True)
    want = rate_history(RACES, prior_mean=0.0, prior_var=1.0,
                        timescale=200.0, return_state=True)
    assert got[1] == want[1] and got[0] == want[0]
    np.testing.assert_array_equal(got[2]["S"], want[2]["S"])
    assert (got[2]["prior_mean"], got[2]["prior_var"],
            got[2]["timescale"]) == (0.0, 1.0, 200.0)


def test_walk_forward_params_still_reach_every_step():
    # walk_forward passes **params on every resume; with the saved values
    # now inherited, explicit drift kwargs must still give the same walk
    a = walk_forward(RACES, warmup=1, timescale=5.0, prior_var=2.0)
    _, _, st = rate_history(RACES[:1], ids=sorted({"a", "b", "c"}),
                            timescale=5.0, prior_var=2.0, return_state=True)
    p, _ = predict_race(st, RACES[1]["runners"], t=RACES[1]["t"])
    np.testing.assert_array_equal(a["records"][0]["p_model"], p)
