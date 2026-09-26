"""State prices stay exhaustive at boundary offsets.

#292. The classic lattice builds the winner field with `shifted_cdf`,
which goes through `_low_high` and PINS any offset at or past `L-2` to
the boundary. `implicit_state_prices` then took a fast path for exact
integers and called `integer_shift(cdf, k)` with the RAW k, whose own
clamp is the much wider +/-(m-1).

So a runner could sit in the field at offset `L-2` and be paid at 60.
The returned state prices stopped being exhaustive claims on one race:

    offsets [60, -60]    sum = 1.8835336397
    offsets [60.1, -60.1] sum = 0.9999867465

An epsilon off the integer moved the total by 0.88, because the
non-integer path had been clamped correctly all along. Guarding the
fast path by the interior condition is exactly equivalent where it
applies -- for an interior integer `_low_high` returns `(k, 1), (k, 0)`,
which is that shift with weight one -- and clamps the boundary integers
the same way the field does.

These run the PURE path, which is what CI exercises and what the
python source here decides. The compiled kernel carries the same branch
and needs rebuilding for the fix to reach a caller who has it; the
source change is in `rust/winning/src/lib.rs` and is NOT compiled or
tested by this repo's automation.
"""
import numpy as np
import pytest

from winning import rustconfig
from winning.classic.lattice import (skew_normal_density,
                                     state_prices_from_offsets)


@pytest.fixture(autouse=True)
def _pure():
    was = rustconfig.rust_active()
    rustconfig.use_rust(False)
    yield
    rustconfig.use_rust(was)


@pytest.fixture(scope="module")
def density():
    return skew_normal_density(50, 0.1)


# the lattice's own truncation error at ordinary offsets, measured
ORDINARY = 0.9999867465
TOL = 5e-9


@pytest.mark.parametrize("a", [48.0, 48.5, 49.0, 49.000001, 50.0, 60.0,
                               60.1, 100.0, 1000.0])
def test_state_prices_sum_to_one_at_and_past_the_boundary(density, a):
    p = state_prices_from_offsets(density, [a, -a])
    assert abs(sum(p) - ORDINARY) < TOL, (a, sum(p))


@pytest.mark.parametrize("k", [49.0, 60.0, -49.0, -60.0])
def test_an_integer_offset_is_continuous_in_its_neighbourhood(density, k):
    """The discontinuity is the symptom that names the cause: the
    integer took a different code path from everything around it."""
    eps = 1e-6
    at = sum(state_prices_from_offsets(density, [k, -k]))
    below = sum(state_prices_from_offsets(density, [k - eps, -(k - eps)]))
    above = sum(state_prices_from_offsets(density, [k + eps, -(k + eps)]))
    assert abs(at - below) < TOL, (k, at, below)
    assert abs(at - above) < TOL, (k, at, above)


def test_interior_integers_are_untouched(density):
    """The fast path is kept where it is valid, and must give exactly
    what it always gave. The tolerance is the LATTICE's ordinary
    truncation error on these three-runner fields, measured: the worst
    is 4.6e-4 at k = 0, where the three offsets collide and the dead
    heat carries the mass."""
    for k in (0.0, 1.0, 5.0, 20.0, 40.0, 47.0, -40.0):
        p = state_prices_from_offsets(density, [k, -k, k / 3.0])
        assert abs(sum(p) - 1.0) < 1e-3, (k, sum(p))
        assert (np.asarray(p) >= 0).all()


def test_a_whole_field_at_the_boundary_is_still_a_distribution(density):
    p = state_prices_from_offsets(density, [60.0, 55.0, -60.0, 3.0, -3.0])
    assert abs(sum(p) - 1.0) < 1e-3, sum(p)
    assert (np.asarray(p) >= -1e-15).all()


def test_the_boundary_clamp_is_where_the_field_puts_it(density):
    """Past the clamp every offset IS the same represented
    distribution, so the prices must be identical -- not merely close."""
    a = state_prices_from_offsets(density, [60.0, -60.0])
    b = state_prices_from_offsets(density, [80.0, -80.0])
    c = state_prices_from_offsets(density, [48.5, -48.5])
    assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-15
    assert np.abs(np.asarray(a) - np.asarray(c)).max() < 1e-15
