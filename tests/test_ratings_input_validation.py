"""Input contracts at the ratings boundaries (#808, #809, #810, #811, #837).

Each block has a regression that failed on main and, next to it, the
ordinary inputs nearby, which must be untouched.
"""
import numpy as np
import pytest
from scipy import sparse

from winning.ratings.factor_ratings import (
    design_loglik, factor_loglik, fit_design_ratings, fit_factor_ratings)


# --- #808: non-finite features, covariates and ridge -----------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("as_sparse", [False, True])
def test_808_non_finite_design_raises(bad, as_sparse):
    Z = np.array([[bad], [0.0], [0.0]])
    if as_sparse:
        Z = sparse.csr_matrix(Z)
    with pytest.raises(ValueError, match="non-finite"):
        fit_design_ratings([(Z, [0])], 1)
    with pytest.raises(ValueError, match="non-finite"):
        design_loglik(np.array([1.0]), [(Z, [0])])


@pytest.mark.parametrize("x", [[np.nan], [[0.0], [np.inf], [1.0]]])
def test_808_non_finite_covariates_raise(x):
    events = [([0, 1, 2], [0], x)]
    with pytest.raises(ValueError, match="non-finite"):
        fit_factor_ratings(events, 3, 1)
    with pytest.raises(ValueError, match="non-finite"):
        factor_loglik(np.zeros((3, 1)), events)


@pytest.mark.parametrize("ridge", [np.nan, np.inf, [1.0, np.nan]])
def test_808_non_finite_ridge_raises(ridge):
    with pytest.raises(ValueError, match="finite"):
        fit_design_ratings([(np.eye(2), [0])], 2, ridge=ridge)


def test_808_ordinary_design_and_factor_fits_unchanged():
    rng = np.random.default_rng(0)
    events = [(rng.normal(size=(4, 3)), list(rng.permutation(4)))
              for _ in range(20)]
    theta, info = fit_design_ratings(events, 3, ridge=[1.0, 0.0, 2.0],
                                     return_info=True)
    assert info["success"] and np.all(np.isfinite(theta))
    # a sparse design with explicit zeros is still accepted
    Zs = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.isfinite(design_loglik(np.array([0.3, -0.3]), [(Zs, [0])])).all()
    fevents = [([0, 1, 2], [2, 0], [1.0, 0.5]) for _ in range(5)]
    B = fit_factor_ratings(fevents, 3, 2, ridge=1.0)
    assert np.all(np.isfinite(B))


# --- #809: cardinal and market noise variances ----------------------------

def _market_args():
    m = np.array([0.2, -0.4, 0.7])
    v = np.array([0.5, 1.0, 2.0])
    y = np.array([1.0, -0.5, 0.2])
    y = y - y.mean()
    return m, v, (lambda _: y)


@pytest.mark.parametrize("kw", [{"beta2": np.nan}, {"beta2": np.inf},
                                {"beta2": [-1.0, 1.0, 1.0]},
                                {"meas_var": np.inf}, {"meas_var": -1.0}])
def test_809_invalid_cardinal_noise_raises(kw):
    from winning.ratings.history import rate_history, update_margins_full
    from winning.ratings.teams import update_team_margins_full
    with pytest.raises(ValueError, match="noise variance"):
        update_margins_full(np.zeros(3), np.eye(3), scores=[1.0, 0.0, -1.0],
                            lengths_scale=1.0, **kw)
    with pytest.raises(ValueError, match="noise variance"):
        update_team_margins_full(np.zeros(3), np.eye(3), np.eye(3),
                                 scores=[1.0, 0.0, -1.0], **kw)
    with pytest.raises(ValueError, match="noise variance"):
        rate_history([{"t": 0.0, "runners": ["a", "b", "c"],
                       "scores": [1.0, 0.0, -1.0]}], **kw)


@pytest.mark.parametrize("tau2", [-0.1, np.nan, np.inf, [-1.0, 1.0, 1.0]])
def test_809_invalid_market_tau2_raises(tau2):
    from winning.ratings.full import update_market_full
    from winning.ratings.market import update_market
    from winning.ratings.teams import update_team_market_full
    m, v, inv = _market_args()
    with pytest.raises(ValueError, match="tau2"):
        update_market(m, v, None, tau2=tau2, invert=inv)
    with pytest.raises(ValueError, match="tau2"):
        update_market_full(m, np.diag(v), None, tau2=tau2, invert=inv)
    with pytest.raises(ValueError, match="tau2"):
        update_team_market_full(m, np.diag(v), np.eye(3), None, tau2=tau2,
                                invert=inv)


def test_809_valid_noise_unchanged_and_zero_legal():
    from winning.ratings.history import update_margins_full
    from winning.ratings.market import update_market
    from winning.ratings.teams import update_team_market_full
    mp, Sp, lz = update_margins_full(np.zeros(3), np.eye(3),
                                     scores=[1.0, 0.0, -1.0],
                                     lengths_scale=1.0)
    assert np.allclose(mp, [0.5, 0.0, -0.5]) and np.isclose(lz, -3.03102425)
    # heterogeneous positive tau2 and zero meas_var / beta2 entries pass
    m, v, inv = _market_args()
    for tau2 in ([0.1, 0.3, 0.5], 0.25):
        out = update_market(m, v, None, tau2=tau2, invert=inv)
        assert np.all(np.isfinite(out[0]))
        out = update_team_market_full(m, np.diag(v), np.eye(3), None,
                                      tau2=tau2, invert=inv)
        assert np.all(np.isfinite(out[0]))
    update_margins_full(np.zeros(3), np.eye(3), scores=[1.0, 0.0, -1.0],
                        beta2=[0.0, 1.0, 1.0], meas_var=0.0)


def test_809_zero_noise_variance_is_legal():
    # an exact observation is a limit, not an invalid variance (#778),
    # so the contract lets zero through; what the market update then
    # does at tau2 = 0 is #778's business
    from winning.ratings._checks import as_noise_variances
    assert np.array_equal(as_noise_variances(0.0, 3, "tau2"), np.zeros(3))
    assert np.array_equal(as_noise_variances([0.0, 2.0], 2, "beta2"),
                          [0.0, 2.0])


# --- #810: tuning selectors -------------------------------------------------

def test_810_order_must_be_a_permutation_of_the_scores():
    from winning.ratings import select
    scores = {1.0: 0.50, 2.0: 0.10, 3.0: 0.40}
    with pytest.raises(ValueError, match="omits"):
        select(scores, "lam", order=[1.0, 3.0])          # drops the optimum
    with pytest.raises(ValueError, match="twice"):
        select(scores, "lam", order=[1.0, 2.0, 3.0, 3.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_810_non_finite_scores_raise(bad):
    from winning.ratings import select, select_grid
    with pytest.raises(ValueError, match="finite"):
        select({1.0: bad, 2.0: 0.1, 3.0: 0.4}, "lam")
    with pytest.raises(ValueError, match="finite"):
        select_grid({("bad", 0): bad, ("good", 0): 0.5, ("worse", 0): 0.8},
                    ["model", "axis"])


def test_810_complete_orders_and_finite_scores_unchanged():
    from winning.ratings import select, select_grid
    scores = {1.0: 0.50, 2.0: 0.10, 3.0: 0.40}
    best, rep = select(scores, "lam", order=[3.0, 2.0, 1.0])
    assert best == 2.0 and not rep.at_edge and rep.spread == 0.4
    best, reps = select_grid({("a", 0): 0.5, ("b", 0): 0.3, ("b", 1): 0.4},
                             ["model", "axis"])
    assert best == ("b", 0) and reps["model"].best == "b"


def test_810_sweeps_with_a_repeated_grid_value_still_select():
    # the sweeps collapse a repeated value in their score dict, so they
    # hand select a duplicate-free order (main's result, no late raise)
    from winning.ratings.factor_ratings import sweep_offset_ridge
    from winning.ratings.tracker import tune_block_rho
    ev = [([0, 1, 2], [0, 1], [1.0, 0.5]), ([2, 0, 1], [1], [1.0, -0.5])]
    out = sweep_offset_ridge(ev, ev, 3, 2, offset_grid=(1.0, 3.0, 3.0),
                             n_iter=20)
    assert len(out["val_loss"]) == 3
    contests = [{"runners": ["a", "b", "c"], "t": float(i), "winner": i % 3,
                 "groups": [0, 0, 1]} for i in range(4)]
    out = tune_block_rho(contests, rho_grid=(0.0, 0.2, 0.2))
    assert len(out["evidence"]) == 3


# --- #811: duplicate runner ids within one contest ---------------------------

@pytest.mark.parametrize("kw", [{"winner": 0}, {"winner": 1},
                                {"order": [0, 1, 2]},
                                {"scores": [0.0, 2.0, 1.0]},
                                {"prices": [0.4, 0.4, 0.2]}])
def test_811_tracker_refuses_duplicate_ids_and_keeps_state(kw):
    from winning.ratings.tracker import AbilityTracker
    trk = AbilityTracker(drift=0.0)
    trk.observe(["a", "b"], t=0.0, winner=0)
    before = (dict(trk.state), trk.evidence)
    with pytest.raises(ValueError, match="repeats 'a'"):
        trk.observe(["a", "a", "b"], t=1.0, **kw)
    assert (dict(trk.state), trk.evidence) == before
    with pytest.raises(ValueError, match="repeats 'a'"):
        trk.predict(["a", "a", "b"], t=1.0)


@pytest.mark.parametrize("obs", [{"winner": 0}, {"order": [0, 1, 2]},
                                 {"scores": [0.0, 2.0, 1.0]},
                                 {"p_market": [0.4, 0.4, 0.2]}])
def test_811_history_refuses_duplicate_runners(obs):
    from winning.ratings.history import predict_race, rate_history
    with pytest.raises(ValueError, match="repeats 'a'"):
        rate_history([{"runners": ["a", "a", "b"], "t": 0.0, **obs}])
    state = rate_history([{"runners": ["a", "b"], "t": 0.0, "winner": 0}],
                         return_state=True)[2]
    with pytest.raises(ValueError, match="repeats 'a'"):
        predict_race(state, ["a", "a", "b"])


def test_811_history_refuses_duplicate_population_ids():
    from winning.ratings.history import rate_history
    with pytest.raises(ValueError, match="repeats 'a'"):
        rate_history([{"runners": ["a", "b"], "t": 0.0, "winner": 0}],
                     ids=["a", "a", "b"])
    r, _ = rate_history([{"runners": ["a", "b"], "t": 0.0, "winner": 0}],
                        ids=iter(["a", "b"]))
    assert set(r) == {"a", "b"}


def test_811_factor_form_refuses_duplicate_entities():
    from winning.ratings.factor_ratings import factor_design, predict_factor
    event = [([0, 0, 1], [0, 1, 2], [1.0])]
    with pytest.raises(ValueError, match="repeats 0"):
        factor_design([0, 0, 1], [1.0], 2, 1)
    with pytest.raises(ValueError, match="repeats 0"):
        fit_factor_ratings(event, n_entities=2, n_cov=1)
    with pytest.raises(ValueError, match="repeats 0"):
        factor_loglik(np.zeros((2, 1)), event)
    with pytest.raises(ValueError, match="repeats 0"):
        predict_factor(np.zeros((2, 1)), [0, 0, 1], [1.0])


def test_811_distinct_ids_unchanged():
    from winning.ratings.history import rate_history
    from winning.ratings.tracker import AbilityTracker
    trk = AbilityTracker(drift=0.0)
    trk.observe(["a", "b", "c"], t=0.0, winner=0)
    assert trk.rating("a")[0] > 0 > trk.rating("b")[0]
    r, lz = rate_history([{"runners": ["a", "b", "c"], "t": 0.0,
                           "winner": 0}])
    assert r["a"][0] > 0 and np.isfinite(lz)
