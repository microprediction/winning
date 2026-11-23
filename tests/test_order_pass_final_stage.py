"""The last stage of _order_pass's adjoint sweep divided without the
underflow guard every earlier stage has: an order whose final finisher
is far ahead of the field raised ZeroDivisionError from order_loglik and,
through a far factor node, from update_order_correlated, whose docstring
promises such orders degrade and never raise."""
import numpy as np

from winning.ratings.nway import _order_pass_log, order_loglik, update_order_correlated

M = np.array([-27.0, -21.9, 26.1, -34.8])
SD = np.array([1.33, 1.30, 1.13, 1.15])
ORDER = [3, 0, 1, 2]


def test_final_stage_underflow_is_priced_in_logs():
    lp, g = order_loglik(M, SD, ORDER)
    x = np.linspace(-80.0, 60.0, 200001)
    lp_ref, g_ref = _order_pass_log(M[None, :], SD, ORDER, x)
    assert np.isfinite(lp)
    assert abs(lp - lp_ref[0]) < 1e-3 * abs(lp_ref[0])
    np.testing.assert_allclose(g, g_ref[0], rtol=1e-3, atol=1e-6)
    assert abs(g.sum()) < 1e-6          # a common shift leaves P alone


def test_correlated_order_update_survives_far_nodes():
    # Qf = 31: the outer Hermite nodes shift the field by ~30 sd
    m = np.array([-0.22, 0.39, 0.1, -0.82])
    v = np.full(4, 0.5)
    V = np.array([[-4.2], [3.1], [2.4], [-2.2]])
    mp, vp, lz = update_order_correlated(m, v, [2, 0, 3, 1], V, 1.0, Qf=31)
    assert np.all(np.isfinite(mp)) and np.all(vp > 0) and np.isfinite(lz)


def test_batch_final_stage_matches_scalar():
    # the batched pass (update_order_full's factor nodes) divided the
    # last stage unguarded too, and returned zeros for two gradient
    # entries on this field instead of rerunning it in logs
    from winning.ratings.nway import _order_pass_batch
    lp, g = _order_pass_batch(M[None, :], SD, ORDER)
    lp1, g1 = order_loglik(M, SD, ORDER)
    assert abs(lp[0] - lp1) < 1e-9 * abs(lp1)
    np.testing.assert_allclose(g[0], g1, rtol=1e-9, atol=1e-9)
    assert abs(g[0].sum()) < 1e-6
