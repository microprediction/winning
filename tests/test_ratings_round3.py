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


# --- #651: every resolved contrast enters the conjugate evidence

def test_team_margins_keep_a_precise_contrast_beside_a_noisy_one():
    from winning.ratings.teams import update_team_margins_full
    A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    beta2 = np.array([1e-4, 1e-4, 1e8])
    lz = [update_team_margins_full(np.zeros(2), np.eye(2), A, scores=s,
                                   beta2=beta2)[2]
          for s in ([0.0, 0.0, 0.0], [0.01, -0.01, 0.0])]
    want0 = -0.5 * (np.log(1e-4) + np.log(4 / 3 + 1e-4 / 3 + 2e8 / 3)
                    + 2 * np.log(2 * np.pi))
    assert lz[0] == pytest.approx(want0, abs=1e-6)
    assert lz[1] - lz[0] == pytest.approx(-1.0, abs=1e-5)


def test_unsupported_contrast_is_refused():
    from winning.ratings.teams import update_team_margins_full
    A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="support"):
        update_team_margins_full(np.zeros(2), np.eye(2), A,
                                 scores=[1.0, -1.0, 0.0], beta2=0.0)
    # the same contrast held at zero is fine
    update_team_margins_full(np.zeros(2), np.eye(2), A,
                             scores=[0.5, 0.5, 0.0], beta2=0.0)


@pytest.mark.parametrize("c", [1.0, 1e-3, 1e-6, 1e-7, 1e-12, 1e6])
def test_market_evidence_obeys_the_unit_law(c):
    from winning.ratings.full import update_market_full
    from winning.ratings.market import update_market
    y = c * np.array([1.0, 0.0, -1.0])
    want = -0.5 * (1.0 + 2 * np.log(2 * c * c) + 2 * np.log(2 * np.pi))
    a = update_market(np.zeros(3), np.full(3, c * c), None, tau2=c * c,
                      invert=lambda _p: y)
    b = update_market_full(np.zeros(3), np.eye(3) * c * c, None,
                           tau2=c * c, invert=lambda _p: y)
    assert a[2] == pytest.approx(want, abs=1e-9)
    assert b[2] == pytest.approx(want, abs=1e-9)


# --- #660: diagonal updates are equivariant in the ability unit

@pytest.mark.parametrize("c", [1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e8])
def test_diagonal_updates_are_unit_equivariant(c):
    from winning.ratings.market import update_market, update_race
    from winning.ratings.nway import (update_order_correlated,
                                      update_ranking_exact)
    y = np.array([1.0, 0.0, -1.0])
    ref = []
    for s in (1.0, c):
        out = []
        m, v, _ = update_market(np.zeros(3), np.full(3, s * s), None,
                                tau2=s * s, invert=lambda _p: s * y)
        out += [m / s, v / s ** 2]
        m, v, _ = update_race(np.zeros(3), np.full(3, s * s), winner=0,
                              p_market=np.ones(3), tau2=s * s, beta2=s * s,
                              invert=lambda _p: s * y)
        out += [m / s, v / s ** 2]
        m, v = update_ranking_exact(s * np.array([.3, 0, -.2]),
                                    np.full(3, s * s), [2, 0, 1],
                                    beta2=s * s)
        out += [m / s, v / s ** 2]
        m, v, _ = update_order_correlated(s * np.array([.3, 0, -.2]),
                                          np.full(3, s * s), [2, 0, 1],
                                          s * np.array([.5, -.2, .1]),
                                          beta2=s * s)
        out += [m / s, v / s ** 2]
        ref.append(out)
    for a, b in zip(*ref):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9)


def test_known_coordinate_stays_known_on_ranked_path():
    from winning.ratings.nway import update_ranking_exact
    _, v = update_ranking_exact(np.array([.3, 0, -.2]),
                                np.array([1.0, 0.0, 1.0]), [2, 0, 1])
    assert v[1] == 0.0


# --- #652: walk_forward scores score-only contests

def test_walk_forward_scores_score_only_contests():
    from winning.ratings.history import predict_race, rate_history, walk_forward
    races = [
        {"t": 0.0, "runners": ["a", "b"], "scores": [1.0, 0.0]},
        {"t": 1.0, "runners": ["a", "b"], "scores": [0.0, 1.0]},
        {"t": 2.0, "runners": ["a", "b"], "scores": [2.0, 1.0]},
    ]
    out = walk_forward(races, warmup=1, beta2=1.0, lengths_scale=1.0)
    assert out["n_scored"] == 2
    assert [r["winner"] for r in out["records"]] == [1, 0]
    _, _, st = rate_history(races[:1], ids=["a", "b"], beta2=1.0,
                            lengths_scale=1.0, return_state=True)
    p, _ = predict_race(st, ["a", "b"], t=1.0, beta2=1.0)
    assert np.allclose(out["records"][0]["p_model"], p)
    neg = [dict(r, margins=[-x for x in r.pop("scores")]) for r in
           [dict(r) for r in races]]
    out2 = walk_forward(neg, warmup=1, beta2=1.0, lengths_scale=1.0)
    assert [r["winner"] for r in out2["records"]] == [1, 0]
    assert out2["log_loss_model"] == pytest.approx(out["log_loss_model"])


# --- #654: the history clock never runs backwards

def _state_at_10():
    from winning.ratings.history import rate_history
    return rate_history(
        [{"t": 10.0, "runners": ["a", "b"], "scores": [1.0, 0.0]}],
        ids=["a", "b"], timescale=10.0, prior_var=1.0, beta2=1.0,
        lengths_scale=1.0, return_state=True)[2]


def test_stale_race_is_refused_not_rewound():
    from winning.ratings.history import rate_history
    st = _state_at_10()
    stale = {"t": 5.0, "runners": ["a"], "winner": 0}
    with pytest.raises(ValueError, match="older"):
        rate_history([stale], state=st, timescale=10.0, prior_var=1.0)
    with pytest.raises(ValueError, match="older"):
        rate_history([{"t": 12.0, "runners": ["a"], "winner": 0}, stale],
                     state=st, timescale=10.0, prior_var=1.0)
    with pytest.raises(ValueError, match="finite"):
        rate_history([{"t": float("nan"), "runners": ["a"], "winner": 0}],
                     state=st, timescale=10.0, prior_var=1.0)


def test_missing_time_on_resumed_state_means_now_and_split_equals_single():
    from winning.ratings.history import rate_history
    st = _state_at_10()
    _, _, a = rate_history([{"runners": ["a"], "winner": 0}], state=st,
                           timescale=10.0, prior_var=1.0, return_state=True)
    assert a["t"] == 10.0 and np.allclose(a["S"], st["S"])
    races = [{"t": float(t), "runners": ["a", "b"], "scores": [t % 2, 1.0]}
             for t in range(6)]
    kw = dict(timescale=10.0, prior_var=1.0, beta2=1.0, lengths_scale=1.0)
    _, lz, one = rate_history(races, ids=["a", "b"], return_state=True, **kw)
    _, lz1, s1 = rate_history(races[:3], ids=["a", "b"], return_state=True,
                              **kw)
    _, lz2, two = rate_history(races[3:], state=s1, return_state=True, **kw)
    assert np.allclose(one["m"], two["m"]) and np.allclose(one["S"], two["S"])
    assert lz == pytest.approx(lz1 + lz2)
    # equal timestamps remain valid
    rate_history([races[0], races[0]], ids=["a", "b"], **kw)
