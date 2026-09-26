"""A tracker refuses evidence that would poison its state.

A tracker carries state, so bad evidence is not one bad answer -- it is
a corrupted filter. One NaN score made EVERY entity's mean and variance
NaN, a later clean contest did not recover them, and nothing said a
word:

    before NaN:            [0.8, 0.0, -0.8]
    after NaN:             [nan, nan, nan]
    after a CLEAN contest: [nan, nan, nan]

An infinite score and a NaN price did the same. The lengths did raise,
but as "operands could not be broadcast together with shapes", which
names nothing the caller passed.
"""
import numpy as np
import pytest

from winning.ratings.tracker import AbilityTracker

IDS = ["a", "b", "c"]


def _seeded():
    t = AbilityTracker(seed=0)
    t.observe(IDS, t=0.0, scores=[1.0, 2.0, 3.0])
    return t


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_score_is_refused(bad):
    t = _seeded()
    with pytest.raises(ValueError, match="not finite"):
        t.observe(IDS, t=1.0, scores=[1.0, bad, 3.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_price_is_refused(bad):
    t = _seeded()
    with pytest.raises(ValueError, match="not finite"):
        t.observe(IDS, t=1.0, prices=[0.5, bad, 0.2])


def test_the_state_survives_a_refusal():
    """The point of refusing at the door: the filter is still usable."""
    t = _seeded()
    before = [t.rating(i) for i in IDS]
    with pytest.raises(ValueError):
        t.observe(IDS, t=1.0, scores=[1.0, np.nan, 3.0])
    after = [t.rating(i) for i in IDS]
    assert after == before
    # and it keeps working
    t.observe(IDS, t=2.0, scores=[3.0, 1.0, 2.0])
    m = np.array([t.rating(i)[0] for i in IDS])
    v = np.array([t.rating(i)[1] for i in IDS])
    assert np.isfinite(m).all() and np.isfinite(v).all() and (v > 0).all()


def test_the_message_names_the_entry_and_counts_them():
    t = _seeded()
    with pytest.raises(ValueError) as e:
        t.observe(IDS, t=1.0, scores=[np.nan, 2.0, np.nan])
    msg = str(e.value)
    assert "scores[0]" in msg
    assert "2 of 3" in msg


@pytest.mark.parametrize("seq,name", [([1.0, 2.0], "scores"),
                                      ([1.0, 2.0, 3.0, 4.0], "scores")])
def test_a_wrong_length_is_named(seq, name):
    t = _seeded()
    with pytest.raises(ValueError, match="one entry per entrant"):
        t.observe(IDS, t=1.0, scores=seq)


def test_a_wrong_length_price_is_named():
    t = _seeded()
    with pytest.raises(ValueError, match="one entry per entrant"):
        t.observe(IDS, t=1.0, prices=[0.4, 0.3, 0.2, 0.1])


def test_ordinary_evidence_is_untouched():
    """Every valid observer still folds, and the ratings move."""
    t = _seeded()
    t.observe(IDS, t=1.0, scores=[3.0, 1.0, 2.0])
    t.observe(IDS, t=2.0, prices=[0.5, 0.3, 0.2])
    t.observe(IDS, t=3.0, order=[2, 0, 1])
    t.observe(IDS, t=4.0, winner=1)
    m = np.array([t.rating(i)[0] for i in IDS])
    v = np.array([t.rating(i)[1] for i in IDS])
    assert np.isfinite(m).all() and np.isfinite(v).all() and (v > 0).all()
    assert np.abs(m).max() > 0.0


# --- the dense filter has the same hole, and the same fix -------------
#
# rate_history does not carry state ACROSS calls the way the tracker
# does, but it carries it across the history it is given: one
# non-finite margin made every entity's mean and covariance NaN for the
# rest of that history, silently. Both walk_forward variants route
# their evidence through these two doors, so they inherit the contract
# rather than needing their own.

from winning.ratings.history import rate_history, walk_forward  # noqa: E402
from winning.ratings.tracker import walk_forward as tracker_walk_forward  # noqa: E402


def _races(n=4):
    return [{"runners": ["a", "b", "c"], "t": float(i),
             "margins": [0.0, 1.0, 2.0]} for i in range(n)]


@pytest.mark.parametrize("key", ["margins", "scores", "p_market"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_dense_filter_refuses_non_finite_evidence(key, bad):
    races = _races()
    races[2] = {"runners": ["a", "b", "c"], "t": 2.0, key: [0.5, bad, 0.2]}
    with pytest.raises(ValueError, match="not finite"):
        rate_history(races)


def test_the_dense_filter_names_the_argument_and_the_race_length():
    races = _races()
    races[2] = {"runners": ["a", "b", "c"], "t": 2.0, "margins": [0.0, 1.0]}
    with pytest.raises(ValueError, match="one entry per runner"):
        rate_history(races)


def test_a_clean_history_is_unaffected():
    out = rate_history(_races())
    arr = out[0] if isinstance(out, tuple) else out
    vals = list(arr.values()) if isinstance(arr, dict) else list(arr)
    flat = np.array([float(x) for v in vals
                     for x in (v if np.ndim(v) else [v])], dtype=float)
    assert np.isfinite(flat).all()


def test_both_walk_forwards_inherit_the_contract():
    """Neither needs its own check: one routes through rate_history and
    the other through observe, which is the point of fixing the door."""
    dense_bad = _races(5)
    dense_bad[3] = {"runners": ["a", "b", "c"], "t": 3.0,
                    "margins": [0.0, np.nan, 2.0]}
    with pytest.raises(ValueError, match="not finite"):
        walk_forward(dense_bad, warmup=1)

    trk = [{"runners": ["a", "b", "c"], "t": float(i),
            "scores": [0.0, 1.0, 2.0], "winner": 0} for i in range(6)]
    assert tracker_walk_forward(trk, warmup=1)["n_scored"] == 5
    trk[4] = {"runners": ["a", "b", "c"], "t": 4.0,
              "scores": [0.0, np.nan, 2.0], "winner": 0}
    with pytest.raises(ValueError, match="not finite"):
        tracker_walk_forward(trk, warmup=1)


# --- finite input must not leave a filter non-finite either (#300) ----
#
# The finiteness checks above look at the ENTRIES, and for a market law
# like [1e308, 1e308, 1e308] every entry is finite. The sum is not:
# `target / target.sum()` in abilities_from_race divided by inf and
# gave NaN, and both filters then stored NaN means and NaN evidence
# from input they had just accepted, correctly, as finite.
#
# A market law is defined up to a positive factor, so [1e308]*3 is the
# SAME law as [1]*3 and must give the same answer.

@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e307, 1e308, 1.7e308])
def test_a_rescaled_market_law_is_the_same_law(scale):
    ref = AbilityTracker(seed=0)
    ref.observe(IDS, t=0.0, prices=[1.0, 1.0, 1.0])
    want = [ref.rating(i) for i in IDS]

    t = AbilityTracker(seed=0)
    t.observe(IDS, t=0.0, prices=[scale] * 3)
    got = [t.rating(i) for i in IDS]
    assert np.isfinite([g[0] for g in got]).all(), (scale, got)
    assert np.isfinite([g[1] for g in got]).all(), (scale, got)
    assert np.isfinite(t.evidence), (scale, t.evidence)
    assert got == want, (scale, got, want)


@pytest.mark.parametrize("scale", [1e307, 1e308])
def test_the_dense_filter_survives_it_too(scale):
    ratings, evidence = rate_history(
        [{"runners": ["a", "b", "c"], "t": 0.0, "p_market": [scale] * 3}])
    means = np.array([ratings[i][0] for i in ["a", "b", "c"]])
    assert np.isfinite(means).all(), (scale, means)
    assert np.isfinite(evidence), (scale, evidence)


def test_an_unequal_law_rescales_too():
    """Not just the symmetric case: the contrasts must be unchanged."""
    base = [0.5, 0.3, 0.2]
    ref = AbilityTracker(seed=0)
    ref.observe(IDS, t=0.0, prices=base)
    want = [ref.rating(i)[0] for i in IDS]
    for scale in (1e-200, 1e200, 1e308):
        t = AbilityTracker(seed=0)
        t.observe(IDS, t=0.0, prices=[p * scale for p in base])
        got = [t.rating(i)[0] for i in IDS]
        assert np.isfinite(got).all(), (scale, got)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), (scale, got, want)
