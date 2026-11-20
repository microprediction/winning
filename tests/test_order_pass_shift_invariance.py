"""An intermediate stage of the linear-domain order pass can lose
precision while staying above _ORDER_TINY, so no underflow guard sends
it to the log domain: on these far-apart fields (log P ~ -740 and -800)
a gradient entry came back 0 instead of 14.07, or the last one off by
0.25 (#1079). The exact gradient sums to zero (P depends on the means
only through their differences); a sum that breaks this now reruns the
row in logs, in the scalar and the batched pass alike."""
import numpy as np
import pytest

from winning.ratings.nway import _order_pass, _order_pass_batch, _order_pass_log

CASES = [
    ([23.141, 1.712, -32.369, -6.761], [1.384, 1.296, 0.787, 0.944], [2, 3, 0, 1]),
    ([28.034, 8.505, 0.338, -7.7, -18.405], [0.58, 0.628, 0.889, 1.23, 1.086],
     [4, 1, 0, 3, 2]),
]


@pytest.mark.parametrize("m, sd, order", CASES)
def test_far_field_gradient_matches_log_pass(m, sd, order):
    m, sd = np.array(m), np.array(sd)
    x = np.linspace(m.min() - 8 * sd.max(), m.max() + 8 * sd.max(), 2001)
    lp_ref, g_ref = _order_pass_log(m[None, :], sd, order, x)
    # the batched pass sizes its own lattice, so
    # the tolerances allow for the discretisation, not for the failure
    for lp, g in (_order_pass(m, sd, order),
                  tuple(a[0] for a in _order_pass_batch(m[None, :], sd, order))):
        assert abs(lp - lp_ref[0]) < 1e-6 * abs(lp_ref[0])
        np.testing.assert_allclose(g, g_ref[0], rtol=1e-3, atol=1e-2)
        assert abs(g.sum()) < 1e-6


def test_ordinary_fields_never_take_the_log_path(monkeypatch):
    # the check must not fire on ordinary inputs: defaults stay bitwise
    import winning.ratings.nway as nway
    calls = []
    real = nway._order_pass_log
    monkeypatch.setattr(nway, "_order_pass_log",
                        lambda *a: calls.append(1) or real(*a))
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(3, 8))
        m = rng.normal(scale=2.0, size=n)
        sd = rng.uniform(0.5, 1.5, n)
        order = [int(i) for i in rng.permutation(n)]
        _order_pass(m, sd, order)
        _order_pass_batch(np.vstack([m, m + rng.normal(size=n)]), sd, order)
    assert not calls
