"""The behavioural half of port parity, in python alone.

`parity/check_divergence.py` runs the same malformed inputs through
python, R, julia and the browser and fails when they disagree about
ACCEPTING them. It needs four toolchains, so CI runs it in the parity
job; these tests pin the python side of every divergence it found, so
a regression is caught by the ordinary suite too.

Every one of these was a real divergence on main, and none had been
reported -- value parity (`check_vectors.py` and friends) cannot see
them, because a port that quietly recycles a short `D` agrees with
everyone on every well-formed vector and still prices a different race.
"""
import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities

MU = np.array([0.0, 0.3, -0.2, 0.5])
D4 = np.ones(4)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_ability_is_refused(bad):
    """python and the browser returned NaN probabilities; R and julia
    refused. mu was the one argument nobody checked."""
    mu = MU.copy()
    mu[1] = bad
    with pytest.raises(ValueError, match="not finite"):
        race_probabilities(mu, D=D4)


def test_an_empty_field_is_refused():
    with pytest.raises(ValueError, match="empty"):
        race_probabilities(np.array([]), D=np.array([]))


def test_the_message_names_the_entry_and_counts_them():
    mu = np.array([np.nan, 0.3, np.nan, 0.5])
    with pytest.raises(ValueError) as e:
        race_probabilities(mu, D=D4)
    msg = str(e.value)
    assert "mu[0]" in msg and "2 of 4" in msg


def test_a_single_runner_is_still_a_race():
    p = np.asarray(race_probabilities(np.array([0.5]), D=np.array([1.0])))
    assert p.shape == (1,) and abs(p[0] - 1.0) < 1e-12


def test_the_documented_spellings_all_still_price():
    """The scan also found R and julia REFUSING the scalar V that
    as_loadings documents; python was right there and must stay right."""
    want = np.asarray(race_probabilities(MU, D=D4))
    for V in (0.4, None):
        p = np.asarray(race_probabilities(MU, V=V, D=D4))
        assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9
        if V is None:
            assert np.abs(p - want).max() == 0.0
    # a scalar D is the same variance for everyone
    assert np.abs(np.asarray(race_probabilities(MU, D=1.0)) - want).max() == 0.0


def test_the_inverse_keeps_its_own_contract():
    p = np.asarray(abilities_from_race(np.array([0.4, 0.3, 0.2, 0.1]), D=D4))
    assert p.shape == (4,) and np.isfinite(p).all()
    # R answered a two-runner race from a four-entry D; python refuses
    with pytest.raises(ValueError):
        abilities_from_race(np.array([0.5, 0.5]), D=D4)


def test_the_case_file_is_wired_to_the_runner():
    """A scan that silently stops covering things passes forever."""
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = json.load(open(os.path.join(root, "parity",
                                        "divergence_cases.json")))["cases"]
    assert len(cases) >= 25
    verbs = {c["verb"] for c in cases}
    assert {"race", "inverse", "topk"} <= verbs
    for name in ("divergence_py.py", "divergence_js.mjs", "divergence_r.R",
                 "divergence_jl.jl", "check_divergence.py"):
        assert os.path.exists(os.path.join(root, "parity", name)), name
