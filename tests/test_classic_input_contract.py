"""The classic boundary refuses what is not a distribution (#339) and
reads target prices as relative mass (#377)."""
import numpy as np
import pytest

from winning import rustconfig
from winning.classic.lattice import (as_classic_density, skew_normal_density,
                                     state_prices_from_extended_offsets,
                                     state_prices_from_offsets)
from winning.classic.lattice_calibration import (solve_for_implied_offsets,
                                                 state_price_implied_ability)


@pytest.fixture(autouse=True)
def _pure():
    was = rustconfig.rust_active()
    rustconfig.use_rust(False)
    yield
    rustconfig.use_rust(was)


@pytest.fixture(scope="module")
def density():
    return np.asarray(skew_normal_density(50, 0.1))


# --- #339 ---------------------------------------------------------------

def test_raw_counts_are_the_same_law_as_frequencies(density):
    offsets = [-3, 0.5, 2]
    p = state_prices_from_offsets(density, offsets)
    q = state_prices_from_offsets(10 * density, offsets)
    np.testing.assert_allclose(q, p, atol=1e-15)
    assert abs(sum(p) - 1) < 1e-13 and min(p) >= 0


@pytest.mark.parametrize("bad, msg", [
    ([0.0] * 7, "no positive mass"),
    ([0.2, -0.1, 0.9, 0, 0, 0, 0], "negative atom"),
    ([0.25] * 8, "odd length"),
    ([0.1, np.nan, 0.9, 0, 0, 0, 0], "non-finite"),
    ([0.1, np.inf, 0.9, 0, 0, 0, 0], "non-finite"),
    ([0.25, 0.5, 0.25], "L >= 3"),
    ([0.2, 0.2, 0.2, 0.2, 0.2], "L >= 3"),
    ([], "nonempty"),
])
def test_forward_and_inverse_refuse_a_non_distribution(bad, msg):
    with pytest.raises(ValueError, match=msg):
        state_prices_from_offsets(bad, [0, 0])
    with pytest.raises(ValueError, match=msg):
        solve_for_implied_offsets([0.7, 0.3], bad)
    with pytest.raises(ValueError, match=msg):
        state_prices_from_extended_offsets(bad, [0.0, 1.0])


def test_roundoff_negatives_are_clipped_not_refused():
    d = np.array([0, 0, 0.25, 0.5, 0.25, 0, -1e-17])
    out = as_classic_density(d)
    assert out.min() == 0.0 and out.sum() == pytest.approx(1.0)


def test_smallest_lattice_round_trips():
    d = np.array([0, 0, 0.25, 0.5, 0.25, 0, 0])      # L = 3
    target = state_prices_from_offsets(d, [0, 0])
    a = solve_for_implied_offsets(target, d, implied_offsets_guess=[0, 0])
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(state_prices_from_offsets(d, list(a)), target,
                               atol=1e-12)


def test_empty_or_nonfinite_offset_samples_are_refused(density):
    with pytest.raises(ValueError, match="empty"):
        solve_for_implied_offsets([0.5, 0.5], density, offset_samples=[])
    with pytest.raises(ValueError, match="non-finite"):
        solve_for_implied_offsets([0.5, 0.5], density, offset_samples=[1.0, np.nan])


# --- #377 ---------------------------------------------------------------

def _gauge_free(a):
    a = np.asarray(a, float)
    return a - a.mean()


@pytest.mark.parametrize("c", [1.1, 2.0, 10.0, 0.5])
def test_inverse_is_invariant_to_target_scale(c):
    d = skew_normal_density(500, 0.01, a=1.5)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    a1 = solve_for_implied_offsets(p, d, implied_offsets_guess=[0] * 4)
    ac = solve_for_implied_offsets(c * p, d, implied_offsets_guess=[0] * 4)
    np.testing.assert_allclose(_gauge_free(ac), _gauge_free(a1), atol=1e-12)
    back = np.asarray(state_prices_from_offsets(d, list(ac)))
    assert np.abs(back - p).max() < 1e-5


@pytest.mark.parametrize("bad, msg", [
    ([0.5, -0.1, 0.6], "negative"),
    ([0.0, 0.0], "no positive mass"),
    ([0.5, np.nan], "non-finite"),
    ([0.5, np.inf], "non-finite"),
])
def test_inverse_refuses_a_target_that_is_not_relative_mass(bad, msg):
    d = skew_normal_density(50, 0.1)
    with pytest.raises(ValueError, match=msg):
        state_price_implied_ability(bad, d)
