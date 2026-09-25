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
