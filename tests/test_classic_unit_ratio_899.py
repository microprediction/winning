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
