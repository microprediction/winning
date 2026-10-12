"""Ratings-filter regressions, round 3: shape contracts, contrast-space
evidence, unit equivariance, ordered-statistics tails and history
bookkeeping."""

import numpy as np
import pytest

from winning.ratings.full import update_order_full
from winning.ratings.nway import update_order_correlated
from winning.ratings.teams import update_team_order_full


# --- #456: singleton orders validate shapes before the tautology return

M3 = np.array([0.0, 1.0, 2.0])
S3 = np.eye(3)


@pytest.mark.parametrize("call", [
    lambda: update_order_full(M3, np.array([1.0, 2.0]), [1]),
    lambda: update_order_full(M3, S3, [1], V=np.ones((2, 2))),
    lambda: update_order_full(M3, S3, [1], beta2=np.ones(2)),
    lambda: update_order_correlated(M3, np.ones(3), [1], np.ones((2, 2))),
    lambda: update_team_order_full(M3, S3, np.ones((2, 2)), [1]),
    lambda: update_team_order_full(M3, S3, np.ones((2, 3)), [1],
                                   V=np.ones((3, 3))),
])
def test_singleton_order_refuses_malformed_shapes(call):
    with pytest.raises(ValueError):
        call()


def test_singleton_order_is_still_an_exact_identity():
    m, S, lz = update_order_full(M3, S3, [1], V=np.ones((3, 1)))
    assert lz == 0.0 and np.array_equal(m, M3) and np.array_equal(S, S3)
    m, v, lz = update_order_correlated(M3, np.ones(3), [2], np.ones((3, 1)))
    assert lz == 0.0 and np.array_equal(m, M3)
    m, S, lz = update_team_order_full(M3, S3, np.ones((2, 3)), [0])
    assert lz == 0.0 and np.array_equal(S, S3)


# --- #648: an omitted runner cannot move a partial-order likelihood

@pytest.mark.parametrize("om,osd", [(0.0, 1.0), (0.0, 1e4), (1e6, 1.0),
                                    (-1e6, 1.0), (0.0, 1e6)])
def test_partial_order_ignores_omitted_runner(om, osd):
    from winning.ratings.nway import _order_pass_batch, order_loglik
    m = np.array([.4, .1, -.2])
    sd = np.array([1., .8, 1.2])
    ll0, g0 = order_loglik(m, sd, [0, 1, 2])
    assert abs(ll0 - (-1.2306349818290565)) < 1e-6
    for pos in range(4):
        mm, ss = np.insert(m, pos, om), np.insert(sd, pos, osd)
        o = [i if i < pos else i + 1 for i in range(3)]
        ll, g = order_loglik(mm, ss, o)
        lb, gb = _order_pass_batch(mm[None, :], ss, o)
        assert ll == pytest.approx(ll0, abs=1e-12) and ll <= 0.0
        assert lb[0] == pytest.approx(ll0, abs=1e-9)
        assert np.allclose(np.delete(g, pos), g0, atol=1e-10)
        assert g[pos] == 0.0 and gb[0, pos] == 0.0


# --- #655: contradicted orders keep a full, gauge-consistent score

@pytest.mark.parametrize("E", [45.0, 50.0])
def test_contradicted_order_score_is_complete(E):
    from winning.ratings.nway import (_order_pass_batch, order_loglik,
                                      update_ranking_exact)
    m = np.linspace(-E, E, 4)
    sd = np.sqrt(np.full(4, 2.0))
    order = np.arange(4)
    ll, g = order_loglik(m, sd, order, L=4001)
    assert np.isfinite(ll) and ll < -1000
    assert abs(g.sum()) < 1e-8 * np.abs(g).max()       # translation gauge
    eps = 1e-2
    fd = [(order_loglik(m + e, sd, order, L=4001)[0]
           - order_loglik(m - e, sd, order, L=4001)[0]) / (2 * eps)
          for e in np.eye(4) * eps]
    assert np.allclose(g, fd, rtol=2e-4)
    mp, vp = update_ranking_exact(m, np.ones(4), order, beta2=1.0)
    assert np.all(np.abs(mp - m) > 1.0) and np.all(vp < 0.9)
    lb, gb = _order_pass_batch(m[None, :], sd, order)
    assert np.isfinite(lb[0]) and np.all(gb[0] != 0.0)
    assert abs(gb[0].sum()) < 1e-8 * np.abs(gb[0]).max()


def test_log_domain_order_pass_matches_linear_on_ordinary_orders():
    from winning.ratings.nway import _order_pass, _order_pass_log
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = rng.normal(size=6)
        sd = rng.uniform(.5, 2, 6)
        o = rng.permutation(6)
        l1, g1 = _order_pass(m, sd, o)
        x = np.linspace(float((m - 8 * sd.max()).min()),
                        float((m + 8 * sd.max()).max()), 2001)
        l2, g2 = _order_pass_log(m[None], sd, o, x)
        assert abs(l1 - l2[0]) < 1e-12 and np.abs(g1 - g2[0]).max() < 1e-12


# --- #650: a perfectly common belief component changes nothing

def test_common_component_is_invisible_to_full_updates():
    from winning.ratings.full import update_order_full, update_winner_full
    from winning.ratings.teams import update_team_winner_full
    from scipy.special import ndtr
    m = np.array([1e-6, 0.0])
    eps = 1e-12
    a = np.array([1.0, -1.0])
    S0 = eps * np.eye(2)
    S1 = S0 + np.ones((2, 2))
    for fn, kw in [(update_winner_full, {"winner": 0}),
                   (update_order_full, {"order": [0, 1]}),
                   (update_team_winner_full, {"A": np.eye(2), "winner": 0})]:
        out = [fn(m, S, beta2=eps, **kw) for S in (S0, S1)]
        for mp, Sp, lz in out:
            assert np.exp(lz) == pytest.approx(ndtr(0.5), rel=1e-4)
            assert a @ mp == pytest.approx(1.5091604338370335e-6, rel=1e-3)
            assert a @ Sp @ a == pytest.approx(1.4861754e-12, rel=2e-3)


@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6, 1e-8])
def test_common_component_three_runner_order(eps):
    from winning.ratings.full import update_order_full
    a = np.array([1.0, -1.0, 0.0])
    m = np.array([np.sqrt(eps), 0.0, -0.5 * np.sqrt(eps)])
    r = []
    for S in (eps * np.eye(3), eps * np.eye(3) + np.ones((3, 3))):
        mp, Sp, lz = update_order_full(m, S, [0, 2, 1], beta2=eps)
        r.append((lz, a @ mp / np.sqrt(eps), a @ Sp @ a / eps))
    assert np.allclose(r[0], r[1], rtol=1e-3)


def test_psd_repair_keeps_valid_small_eigenvalues():
    from winning.ratings.full import _psd_repair
    S = 1e-12 * np.eye(2) + np.ones((2, 2))
    assert np.array_equal(_psd_repair(S), S)
    neg = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-15]])
    assert np.linalg.eigvalsh(_psd_repair(neg)).min() >= -1e-15


def test_market_full_accepts_a_singular_belief():
    from winning.ratings.full import update_market_full
    y = np.array([1.0, 0.0, -1.0])
    S = np.ones((3, 3))                      # common-mode only: singular
    m, Sp, lz = update_market_full(np.zeros(3), S, None, tau2=1.0,
                                   invert=lambda p: y)
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(Sp))
    assert np.allclose(m, 0.0) and np.allclose(Sp, S)
