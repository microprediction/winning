import numpy as np
import pytest

# Make property tests optional if Hypothesis is not installed
pytest.importorskip("hypothesis")
import hypothesis as h
import hypothesis.strategies as st

from winning.thurstone import AbilityCalibrator
from winning.thurstone.inference import implicit_state_prices


@st.composite
def ability_vectors(draw, n_min=2, n_max=8, lo=-20.0, hi=20.0):
    n = draw(st.integers(n_min, n_max))
    vals = draw(
        st.lists(
            st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return vals


@h.given(ability_vectors())
def test_prices_sum_to_one(base, ability):
    cal = AbilityCalibrator(base)
    p = cal.state_prices_from_ability(ability)
    s = sum(p)
    assert 0.999999 <= s <= 1.000001
    assert all(pi >= 0 for pi in p)


@h.given(ability_vectors().map(lambda a: sorted(a)))
def test_monotone_with_respect_to_offset(base, ability):
    # Lower (left) ability should not hurt the same runner
    cal = AbilityCalibrator(base)
    i = len(ability) // 2
    a1 = ability[:]
    a2 = ability[:]
    a2[i] -= 1.0  # make runner i strictly better
    p1 = cal.state_prices_from_ability(a1)
    p2 = cal.state_prices_from_ability(a2)
    assert p2[i] >= p1[i] - 1e-9  # allow tiny numerical slack


def test_implicit_curve_is_monotone(base):
    # Hold others at 0, vary one runner
    grid = [x / 2 for x in range(-40, 41)]  # -20..+20 steps
    prices = [implicit_state_prices(base, [off, 0.0, 0.0])[0] for off in grid]
    diffs = np.diff(prices)
    # non-increasing up to tiny numerical slack
    assert np.all(diffs <= 1e-8)
