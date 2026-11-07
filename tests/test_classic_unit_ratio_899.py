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
FIELDS = [[-20, -2, 6.5], [-30, -25, 0, 1, 2], [-40, -3, 3, 40]]


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("live", FIELDS)
def test_a_scratched_runner_leaves_the_live_field_at_the_requested_ratio(live, ratio):
    p = np.asarray(price(D, live, unit_ratio=ratio))
    q = np.asarray(price(D, live + [float("inf")], unit_ratio=ratio))
    assert q[-1] == 0.0
    np.testing.assert_allclose(q[:-1], p, atol=1e-15)


def test_the_ratio_is_honoured_not_defaulted():
    live = FIELDS[0]
    assert np.abs(np.asarray(price(D, live, unit_ratio=2)) - np.asarray(price(D, live, unit_ratio=3))).max() > 1e-5
    q = np.asarray(price(D, live + [float("inf")], unit_ratio=2))[:-1]
    assert np.abs(q - np.asarray(price(D, live, unit_ratio=3))).max() > 1e-5   # was identical


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("field", [[-20, -2, 6.5], [-20, -2, 6.5, float("inf")]])
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


def _trace(monkeypatch, field, ratio):
    import winning.classic.lattice as lat
    calls = []
    orig_e, orig_c = lat.state_prices_from_extended_offsets, lat._state_prices_from_clustered_offsets

    def spy_e(density, offsets, max_depth=3, unit_ratio=3):
        calls.append(("extended", bool(np.array_equal(density, D)), list(offsets), unit_ratio))
        return orig_e(density, offsets, max_depth=max_depth, unit_ratio=unit_ratio)

    def spy_c(density, offsets, fast_ndxs, unit_ratio, max_depth):
        calls.append(("clustered", len(offsets), len(fast_ndxs), max_depth, unit_ratio))
        return orig_c(density=density, offsets=offsets, fast_ndxs=fast_ndxs,
                      unit_ratio=unit_ratio, max_depth=max_depth)
    monkeypatch.setattr(lat, "state_prices_from_extended_offsets", spy_e)
    monkeypatch.setattr(lat, "_state_prices_from_clustered_offsets", spy_c)
    out = np.asarray(lat.state_prices_from_extended_offsets(D, field, unit_ratio=ratio))
    return out, calls


@pytest.mark.parametrize("ratio", [2, 4])
def test_both_cluster_recursions_run_on_the_dilated_and_base_lattices(monkeypatch, ratio):
    # a split with slow and fast both non-empty and depth left recurses
    # into the slow group on the dilated density and the fast group on the
    # base one; every one of those calls must carry the ratio
    _, calls = _trace(monkeypatch, [-20, -2, 6.5], ratio)
    splits = [c for c in calls if c[0] == "clustered" and 0 < c[2] < c[1] and c[3] > 0]
    assert splits, calls
    base = {c[1] for c in calls if c[0] == "extended"}
    assert base == {True, False}, calls   # base and dilated densities both priced
    assert all(c[-1] == ratio for c in calls)


@pytest.mark.parametrize("ratio", [2, 4])
def test_a_finite_bad_runner_is_removed_at_the_requested_ratio(monkeypatch, ratio):
    p = np.asarray(price(D, [-20, -2, 6.5], unit_ratio=ratio))
    q, calls = _trace(monkeypatch, [-20, -2, 6.5, 200], ratio)
    # the 200 is rewritten as +inf and the field priced again
    assert any(c[0] == "extended" and c[2][-1] == float("inf") for c in calls[1:]), calls
    assert all(c[-1] == ratio for c in calls)
    assert q[-1] == 0.0
    np.testing.assert_allclose(q[:-1], p, atol=1e-15)


def test_ratio_sensitive_prices_are_pinned():
    # the python, R and browser ports agree at every ratio. [-20, -2, 2]
    # was the field here until #1006: it fits this lattice, so it is now
    # priced exactly and never reaches the ratio; [-20, -2, 6.5] spans 26.5
    # against the 26 the range test allows, and must cluster
    for ratio, want in [(2, [0.9994313990, 0.0005320731, 0.0000365279]),
                        (3, [0.9993836116, 0.0005729348, 0.0000434535]),
                        (4, [0.9971923608, 0.0025548886, 0.0002527506])]:
        for field in ([-20, -2, 6.5], [-20, -2, 6.5, float("inf")], [-20, -2, 6.5, 200]):
            got = np.asarray(price(D, field, unit_ratio=ratio))[:3]
            np.testing.assert_allclose(got, want, atol=1e-9)
