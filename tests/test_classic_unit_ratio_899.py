"""The classic recursion keeps the caller's unit_ratio on every branch (#899).

state_prices_from_extended_offsets dropped a non-default unit_ratio on its
+Inf removal, bad-runner removal, terminal and clustered recursions, so a
ratio-2 field with a scratched runner came back as the ratio-3 prices.
The browser already propagated it (it exposes no ratio publicly, so the
check here is the invariance itself, at two non-default ratios).
"""
import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density, state_prices_from_extended_offsets as price

D = skew_normal_density(L=15, unit=0.2, scale=1, a=1.5)
FIELDS = [[-20, -2, 2], [-30, -25, 0, 1, 2], [-40, -3, 3, 40]]


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("live", FIELDS)
def test_a_scratched_runner_leaves_the_live_field_at_the_requested_ratio(live, ratio):
    p = np.asarray(price(D, live, unit_ratio=ratio))
    q = np.asarray(price(D, live + [float("inf")], unit_ratio=ratio))
    assert q[-1] == 0.0
    np.testing.assert_allclose(q[:-1], p, atol=1e-15)


def test_the_ratio_is_honoured_not_defaulted():
    live = FIELDS[0]
    assert np.abs(np.asarray(price(D, live, unit_ratio=2)) - np.asarray(price(D, live, unit_ratio=3))).max() > 1e-4
    q = np.asarray(price(D, live + [float("inf")], unit_ratio=2))[:-1]
    assert np.abs(q - np.asarray(price(D, live, unit_ratio=3))).max() > 1e-4   # was identical


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("field", [[-20, -2, 2], [-20, -2, 2, float("inf")]])
def test_every_recursive_call_receives_the_requested_ratio(monkeypatch, field, ratio):
    # The scratch comparison above passes if BOTH sides drop the ratio
    # further down; this records the ratio each recursion actually gets.
    # These fields reach the dilated, slow- and fast-cluster recursions at
    # several depths (review of #983).
    import winning.classic.lattice as lat
    seen = []
    orig_e, orig_c = lat.state_prices_from_extended_offsets, lat._state_prices_from_clustered_offsets

    def spy_e(density, offsets, max_depth=3, unit_ratio=3):
        seen.append(("extended", unit_ratio))
        return orig_e(density, offsets, max_depth=max_depth, unit_ratio=unit_ratio)

    def spy_c(density, offsets, fast_ndxs, unit_ratio, max_depth):
        seen.append(("clustered", unit_ratio))
        return orig_c(density=density, offsets=offsets, fast_ndxs=fast_ndxs,
                      unit_ratio=unit_ratio, max_depth=max_depth)
    monkeypatch.setattr(lat, "state_prices_from_extended_offsets", spy_e)
    monkeypatch.setattr(lat, "_state_prices_from_clustered_offsets", spy_c)
    lat.state_prices_from_extended_offsets(D, field, unit_ratio=ratio)
    assert {k for k, _ in seen} == {"extended", "clustered"}
    assert len(seen) > 3 and all(r == ratio for _, r in seen), seen


@pytest.mark.parametrize("bad", [1, 0.6, 0, -2, float("nan"), float("inf"), True, "3"])
def test_a_ratio_that_does_not_coarsen_is_refused(bad):
    # honoured in every recursion, 0.6 priced two runners with positive
    # exact prices at zero (review of #983)
    with pytest.raises(ValueError, match="unit_ratio"):
        price(D, FIELDS[0], unit_ratio=bad)
