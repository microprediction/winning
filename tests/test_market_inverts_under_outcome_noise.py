"""Every market path inverts prices under the OUTCOME model -- its
beta2 and base -- whether or not loadings are present (#94).

Prices generated at race noise beta2 must come back, through each
public path, as the abilities that generated them; that is the
"inverted under the outcome model, race noise only" contract of
CLAUDE.md, now honoured without V too."""

import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities
from winning.ratings.history import rate_history
from winning.ratings.market import update_market, update_race
from winning.ratings.tracker import AbilityTracker

SKILL = np.array([1.2, 0.2, -0.4, -1.0])          # max-wins, centred
IDS = ["a", "b", "c", "d"]


def _prices(beta2, base="normal"):
    return race_probabilities(-SKILL, D=beta2, base=base)


def _direct(p, beta2, tau2, base="normal"):
    inv = lambda q: -abilities_from_race(q, D=beta2, base=base)
    return update_market(np.zeros(4), np.ones(4), p, tau2=tau2, invert=inv)


@pytest.mark.parametrize("beta2", [0.25, 4.0])
def test_update_race_without_V(beta2):
    p = _prices(beta2)
    want = _direct(p, beta2, 0.1)
    got = update_race(np.zeros(4), np.ones(4), p_market=p, beta2=beta2,
                      tau2=0.1)
    assert np.allclose(got[0], want[0], atol=1e-8), (got[0], want[0])
    assert abs(got[2]["logZ_market"] - want[2]) < 1e-8
    # and it is NOT the D=1 inversion
    wrong = _direct(p, 1.0, 0.1)
    assert np.abs(got[0] - wrong[0]).max() > 0.05


def test_update_race_forwards_the_base():
    scale = np.sqrt(6) / np.pi
    p = np.exp(SKILL / scale); p /= p.sum()
    want = update_market(np.zeros(4), np.ones(4), p, tau2=0.25,
                         invert=lambda q: scale * (np.log(q) - np.log(q).mean()))
    got = update_race(np.zeros(4), np.ones(4), p_market=p, tau2=0.25,
                      base="gumbel")
    assert np.allclose(got[0], want[0], atol=1e-4), (got[0], want[0])
    assert abs(got[2]["logZ_market"] - want[2]) < 1e-3


def test_a_numerical_kwarg_keeps_the_loadings():
    V = np.array([[0.5], [0.4], [-0.3], [-0.6]])
    p = race_probabilities(-SKILL, V=V, D=1.0)
    a = update_race(np.zeros(4), np.ones(4), p_market=p, V=V, tau2=0.1)
    b = update_race(np.zeros(4), np.ones(4), p_market=p, V=V, tau2=0.1,
                    points=1001)
    assert np.allclose(a[0], b[0], atol=1e-4), (a[0], b[0])


@pytest.mark.parametrize("beta2", [0.25, 4.0])
def test_tracker_without_groups(beta2):
    p = _prices(beta2)
    want = _direct(p, beta2, 0.1)[0]
    t = AbilityTracker(beta2=beta2, tau2=0.1, init_var=1.0, seed=0)
    t.observe(IDS, t=0.0, prices=p)
    got = np.array([t.rating(i)[0] for i in IDS])
    assert np.allclose(got - got.mean(), want - want.mean(), atol=1e-8)


@pytest.mark.parametrize("beta2", [0.25, 4.0])
def test_dense_history_without_V(beta2):
    p = _prices(beta2)
    want = _direct(p, beta2, 0.1)[0]
    ratings, _ = rate_history([{"t": 0.0, "runners": IDS, "p_market": p}],
                              beta2=beta2, tau2=0.1)
    got = np.array([ratings[i][0] for i in IDS])
    assert np.allclose(got - got.mean(), want - want.mean(), atol=1e-8)
