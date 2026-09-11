"""Factor ratings (winning.ratings.factor_ratings) and block correlation
in the tracker.

The fitter is a MAP through the engine's ranking likelihood, so the
gates are identities rather than posteriors: every event shape's
log-likelihood equals the single-event engine call, the analytic
gradient equals finite differences, and the optimum equals the
reference per-event loop the measurements were made with. The
behavioural claims the spec carries (per-feature ridge is the
mechanism; the offset sweep finds an interior optimum when offsets
exist and the scalar limit when they do not; the contrast report
flags a chosen covariate; evidence selects the block correlation) are
each reproduced on a synthetic world with known truth.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from winning.ratings.factor_ratings import (
    _Stacked, _design_triplets, _loglik_terms, _winner_batch,
    covariate_contrast_report, design_loglik, factor_design, factor_loglik,
    fit_design_ratings, fit_factor_ratings, predict_factor,
    sweep_offset_ridge)
from winning.ratings.nway import (_grad_logp_row, order_loglik,
                                  update_order_correlated,
                                  update_winner_correlated)
from winning.ratings.tracker import (AbilityTracker, block_loadings,
                                     tune_block_rho, walk_forward)
from winning.ratings.simulate import (block_world, factor_world,
                                      mixed_events)


N_FEAT = 7


def _mixed_events(rng):
    return mixed_events(rng, N_FEAT)


@pytest.mark.parametrize("base", ["normal", "logistic"])
def test_event_loglik_matches_single_event_engine_calls(base):
    rng = np.random.default_rng(0)
    evs = _mixed_events(rng)
    theta = rng.normal(size=N_FEAT) * 0.5
    lp = design_loglik(theta, evs, base=base)
    from winning.factor.races import race_probabilities
    for e, (Z, order) in enumerate(evs):
        Zd = Z.toarray() if sparse.issparse(Z) else Z
        mu = Zd @ theta
        if len(order) >= 2:
            ref, _ = order_loglik(mu, np.ones(len(mu)), order, base=base)
        else:
            ref = float(np.log(race_probabilities(
                -mu, D=np.ones(len(mu)), base=base)[int(order[0])]))
        assert abs(lp[e] - ref) < 5e-5, (e, lp[e], ref)


@pytest.mark.parametrize("base", ["normal", "logistic"])
def test_gradient_matches_finite_differences(base):
    rng = np.random.default_rng(1)
    st = _Stacked(_design_triplets(_mixed_events(rng), N_FEAT), N_FEAT)
    theta = rng.normal(size=N_FEAT) * 0.5
    w = rng.random(st.n_events) + 0.5

    def f(th):
        lp, dmu = _loglik_terms(th, st, base)
        return float(w @ lp), np.asarray(st.ZT @ (dmu * w[st.row_event])).ravel()

    _, g = f(theta)
    gfd = np.empty(N_FEAT)
    for j in range(N_FEAT):
        e = np.zeros(N_FEAT); e[j] = 1e-5
        gfd[j] = (f(theta + e)[0] - f(theta - e)[0]) / 2e-5
    assert np.abs(g - gfd).max() < 1e-4 * max(1.0, np.abs(gfd).max())


def test_winner_batch_matches_engine_gradient_row():
    rng = np.random.default_rng(2)
    Ms = rng.normal(size=(3, 4))
    lp, g = _winner_batch(Ms)
    for e in range(3):
        ge, p = _grad_logp_row(Ms[e], np.ones(4), 0)
        assert abs(lp[e] - np.log(p)) < 1e-10
        assert np.abs(g[e] - ge).max() < 1e-10


_factor_world = factor_world


def test_fit_matches_reference_loop_and_both_forms_agree():
    rng = np.random.default_rng(3)
    M, P = 12, 2
    _, fevents = _factor_world(rng, M, 5, 150)
    devents = [(factor_design(s, x, M, P), o) for s, o, x in fevents]
    ridge = np.tile([1.0, 5.0], M)
    theta = fit_design_ratings(devents, M * P, ridge=ridge)
    B = fit_factor_ratings(fevents, M, P, ridge=np.array([1.0, 5.0]))
    assert np.abs(theta.reshape(M, P) - B).max() < 1e-12

    # the per-event loop the measurements were made with (bandits
    # linear_ability_map semantics)
    def ref(th):
        total = 0.5 * float(th @ (ridge * th)); grad = ridge * th
        for Z, o in devents:
            Zd = Z.toarray(); mu = Zd @ th
            lp, dmu = order_loglik(mu, np.ones(len(mu)), o)
            total -= lp; grad -= Zd.T @ dmu
        return total, grad
    res = minimize(ref, np.zeros(M * P), jac=True, method="L-BFGS-B",
                   options={"maxiter": 300})
    assert np.abs(res.x - theta).max() < 1e-4


def test_recovers_levels_and_offsets_from_exogenous_covariates():
    rng = np.random.default_rng(4)
    B, evs = _factor_world(rng, 20, 5, 400, off_sd=0.5)
    Bh = fit_factor_ratings(evs, 20, 2, ridge=np.array([1.0, 3.0]))
    assert np.corrcoef(Bh[:, 0], B[:, 0])[0, 1] > 0.9
    assert np.corrcoef(Bh[:, 1], B[:, 1])[0, 1] > 0.6


def test_per_feature_ridge_is_the_mechanism():
    # levels free, offsets shrunk beats one shared penalty AND the
    # scalar rating on held-out log-loss (the spec's chess result, in
    # a synthetic world with small true offsets)
    rng = np.random.default_rng(0)
    B, tr = _factor_world(rng, 40, 5, 400, off_sd=0.25, n_cov=3)
    va = []
    for _ in range(400):
        sub = rng.choice(40, 5, replace=False)
        x = np.array([1.0, float(rng.random() < 0.5), float(rng.random() < 0.5)])
        perf = B[sub] @ x + rng.normal(size=5)
        va.append((sub, np.argsort(-perf), x))
    flat = fit_factor_ratings(tr, 40, 3, ridge=1.0)
    hier = fit_factor_ratings(tr, 40, 3, ridge=np.array([1.0, 20.0, 20.0]))
    scal = fit_factor_ratings([(s, o, x[:1]) for s, o, x in tr], 40, 1)
    l_flat = -factor_loglik(flat, va).mean()
    l_hier = -factor_loglik(hier, va).mean()
    l_scal = -factor_loglik(scal, [(s, o, x[:1]) for s, o, x in va]).mean()
    assert l_hier < l_flat - 0.02, (l_hier, l_flat)
    assert l_hier < l_scal, (l_hier, l_scal)


def test_offset_sweep_interior_when_offsets_exist_scalar_limit_when_not():
    rng = np.random.default_rng(5)
    grid = (1.0, 10.0, 100.0, 1000.0)
    B, tr = _factor_world(rng, 20, 5, 200, off_sd=0.5, n_cov=2)
    va = []                                       # validation from the SAME truth
    for _ in range(200):
        sub = rng.choice(20, 5, replace=False)
        x = np.array([1.0, float(rng.random() < 0.5)])
        perf = B[sub] @ x + rng.normal(size=5)
        va.append((sub, np.argsort(-perf), x))
    sw = sweep_offset_ridge(tr, va, 20, 2, offset_grid=grid)
    assert tuple(sw["offset_grid"]) == grid
    assert sw["val_loss"].shape == (4,)
    assert not sw["at_scalar_limit"] and sw["offset_ridge"] < 1000.0
    assert sw["B"].shape == (20, 2)

    B0, tr0 = _factor_world(rng, 20, 5, 200, off_sd=0.0, n_cov=2)
    va0 = []
    for _ in range(200):
        sub = rng.choice(20, 5, replace=False)
        x = np.array([1.0, float(rng.random() < 0.5)])
        perf = B0[sub] @ x + rng.normal(size=5)
        va0.append((sub, np.argsort(-perf), x))
    sw0 = sweep_offset_ridge(tr0, va0, 20, 2, offset_grid=grid)
    assert sw0["offset_ridge"] >= 100.0, sw0["val_loss"]


def test_covariate_contrast_report_flags_a_chosen_covariate():
    rng = np.random.default_rng(6)
    _, evs = _factor_world(rng, 12, 5, 150)
    rep = covariate_contrast_report(evs, 12, 2)
    assert np.isnan(rep["ratio"][0])            # the constant level column
    assert 0.5 < rep["ratio"][1] < 1.6           # assigned: near the null
    assert rep["share"].shape == (12, 2) and rep["n_entities_seen"] == 12
    # chosen: entities play only under their own condition
    chosen = []
    for s, o, x in evs:
        flag = float(s[0] % 2)
        s2 = np.array([i for i in range(12) if i % 2 == flag])[:5]
        chosen.append((s2, np.arange(5), np.array([1.0, flag])))
    rep2 = covariate_contrast_report(chosen, 12, 2)
    assert rep2["ratio"][1] > 4.0


@pytest.mark.parametrize("base", ["normal", "logistic"])
def test_predict_is_the_likelihood_model(base):
    rng = np.random.default_rng(7)
    B = rng.normal(size=(4, 2))
    sub = np.array([0, 3]); x = np.array([1.0, 1.0])
    p = predict_factor(B, sub, x, base=base)
    lp = factor_loglik(B, [(sub, [0, 1], x)], base=base)
    assert abs(np.log(p[0]) - lp[0]) < 1e-6
    assert abs(p.sum() - 1.0) < 1e-9


def test_input_validation():
    Z = np.eye(3)
    with pytest.raises(ValueError):
        fit_design_ratings([(Z, [0, 0])], 3)            # repeated row
    with pytest.raises(ValueError):
        fit_design_ratings([(Z, [3])], 3)               # row out of range
    with pytest.raises(ValueError):
        fit_design_ratings([(Z, [0, 1])], 3, ridge=[1.0, 2.0])
    with pytest.raises(ValueError):
        fit_design_ratings([(Z, [0, 1])], 3, weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        fit_factor_ratings([([0, 5], [0, 1], [1.0])], 3, 1)   # entity range
    with pytest.raises(ValueError):
        fit_factor_ratings([([0, 1], [0, 1], [1.0, 0.0])], 3, 1)  # x length
    with pytest.raises(ValueError):
        block_loadings(["a", "a"], 1.0)


# -- block correlation in the tracker ---------------------------------------

def test_block_loadings_are_variance_preserving_with_singletons():
    V, b2 = block_loadings(["x", "x", "y", "y", None, "z"], 0.4, beta2=2.0)
    assert V.shape == (6, 2)
    tot = np.diag(V @ V.T) + b2
    assert np.abs(tot - 2.0).max() < 1e-12       # every marginal keeps beta2
    assert b2[4] == 2.0 and b2[5] == 2.0         # singletons untouched
    assert V[4].sum() == 0.0 and V[5].sum() == 0.0
    assert block_loadings(["x", "y", "z"], 0.4)[0] is None   # no group of two
    assert block_loadings(["x", "x"], 0.0)[0] is None


def test_blocked_observe_matches_engine_and_accumulates_evidence():
    ids = list("abcdef"); groups = ["x", "x", "y", "y", None, "z"]
    order = [0, 2, 1, 4, 3, 5]
    trk = AbilityTracker(rho=0.4, drift=0.0, Qf=5)
    m, v = trk._gather(ids, 0.0)
    V, b2 = block_loadings(groups, 0.4, 1.0)
    trk.observe(ids, 0.0, order=order, groups=groups)
    mm, vv, lz = update_order_correlated(m, v, order, V, beta2=b2, Qf=5)
    got_m = np.array([trk.state[i].mean for i in ids])
    got_v = np.array([trk.state[i].var for i in ids])
    assert np.abs(got_m - mm).max() < 1e-12 and np.abs(got_v - vv).max() < 1e-12
    assert abs(trk.evidence - lz) < 1e-12
    m2, v2 = trk._gather(ids, 1.0)               # drift 0: the stored belief
    assert np.abs(m2 - mm).max() < 1e-12 and np.abs(v2 - vv).max() < 1e-12
    trk.observe(ids, 1.0, winner=2, groups=groups)
    mw, vw, lz2 = update_winner_correlated(mm, vv, 2, V, beta2=b2, Qf=5)
    got_m = np.array([trk.state[i].mean for i in ids])
    got_v = np.array([trk.state[i].var for i in ids])
    assert np.abs(got_m - mw).max() < 1e-12 and np.abs(got_v - vw).max() < 1e-12
    assert abs(trk.evidence - (lz + lz2)) < 1e-12
    # prediction prices the same blocked model and sums to one
    p_blk = trk.predict(ids, 1.0, groups=groups)
    p_ind = trk.predict(ids, 1.0)
    assert abs(p_blk.sum() - 1.0) < 1e-9 and np.abs(p_blk - p_ind).max() > 1e-4


def test_groups_at_rho_zero_and_independent_evidence():
    ids = list("abcde"); order = [3, 0, 4, 1, 2]
    a = AbilityTracker(drift=0.0); b = AbilityTracker(drift=0.0, rho=0.0)
    a.observe(ids, 0.0, order=order)
    b.observe(ids, 0.0, order=order, groups=["x", "x", "y", "y", "y"])
    for i in ids:
        assert a.state[i].mean == b.state[i].mean and a.state[i].var == b.state[i].var
    assert a.evidence == b.evidence
    # the independent order path's evidence is log P(order) at the prior
    lp, _ = order_loglik(np.zeros(5), np.sqrt(4.0 + 1.0) * np.ones(5), order)
    assert abs(a.evidence - lp) < 1e-12
    # winner and scores paths add their own log-evidence
    a.observe(ids, 1.0, winner=1)
    a.observe(ids, 2.0, scores=[0.1, -0.3, 0.2, 0.0, 0.4])
    assert np.isfinite(a.evidence) and a.evidence < lp


def test_blocked_scores_and_prices_paths_run_and_keep_marginals():
    ids = list("abcd"); groups = ["x", "x", "y", "y"]
    trk = AbilityTracker(rho=0.5, drift=0.0)
    trk.observe(ids, 0.0, scores=[-1.0, -0.8, 0.3, 0.5], groups=groups,
                prices=[0.4, 0.3, 0.2, 0.1])
    m = np.array([trk.state[i].mean for i in ids])
    v = np.array([trk.state[i].var for i in ids])
    assert np.isfinite(m).all() and (v > 0).all() and (v < 4.0).all()
    assert m[0] > m[3]                            # best score, best price
    assert np.isfinite(trk.evidence)


_block_world = block_world


def test_tune_block_rho_selects_the_true_correlation_by_evidence():
    contests = _block_world(0)
    sel = tune_block_rho(contests, rho_grid=(0.0, 0.5, 0.85), drift=0.0,
                         init_var=1.0, Qf=5)
    assert sel["rho"] == 0.5 and not sel["at_edge"]
    ev = sel["evidence"]
    assert ev[1] > ev[0] + 1.0 and ev[1] > ev[2] + 1.0


def test_walk_forward_passes_groups_and_returns_evidence():
    contests = _block_world(1, n_contests=12)
    out = walk_forward(contests, warmup=4, rho=0.5, drift=0.0, init_var=1.0,
                       Qf=5, market_arm=False)
    assert out["n_scored"] == 8
    assert np.isfinite(out["log_loss_model"]) and np.isfinite(out["evidence"])
    assert out["evidence"] == out["tracker"].evidence


# -- Laplace standard errors ---------------------------------------------------

def test_penalised_hessian_matches_finite_differences_and_is_spd():
    from winning.ratings.factor_ratings import _as_ridge, _penalised_hessian
    rng = np.random.default_rng(21)
    for base in ("normal", "logistic"):
        evs = mixed_events(rng, N_FEAT)
        st = _Stacked(_design_triplets(evs, N_FEAT), N_FEAT)
        theta = rng.normal(size=N_FEAT) * 0.5
        lam = _as_ridge(1.0, N_FEAT); w = rng.random(st.n_events) + 0.5

        def grad(th):
            lp, dmu = _loglik_terms(th, st, base)
            return lam * th - np.asarray(st.ZT @ (dmu * w[st.row_event])).ravel()
        Hfd = np.empty((N_FEAT, N_FEAT))
        for j in range(N_FEAT):
            e = np.zeros(N_FEAT); e[j] = 1e-4
            Hfd[:, j] = (grad(theta + e) - grad(theta - e)) / 2e-4
        H = _penalised_hessian(theta, st, lam, w, base).toarray()
        assert np.abs(H - Hfd).max() < 1e-3 * np.abs(Hfd).max()
        assert np.abs(H - H.T).max() < 1e-12
        assert np.linalg.eigvalsh(H).min() > 0.5           # ridge 1 plus curvature


def test_laplace_se_shapes_paths_and_calibration():
    from winning.ratings.factor_ratings import design_se, factor_se
    rng = np.random.default_rng(22)
    B, evs = factor_world(rng, 20, 5, 300, off_sd=0.5, n_cov=2)
    ridge = np.array([1.0, 3.0])
    Bh, se, info = fit_factor_ratings(evs, 20, 2, ridge=ridge, return_se=True, return_info=True)
    assert Bh.shape == se.shape == (20, 2) and np.isfinite(se).all() and (se > 0).all()
    assert np.abs(factor_se(Bh, evs, ridge=ridge) - se).max() == 0.0
    # the same through the design form
    dev = [(factor_design(s, x, 20, 2), o) for s, o, x in evs]
    th, se_d = fit_design_ratings(dev, 40, ridge=np.tile(ridge, 20), return_se=True)
    assert np.abs(se_d.reshape(20, 2) - se).max() < 1e-8
    assert np.abs(design_se(th, dev, 40, ridge=np.tile(ridge, 20)) - se_d).max() == 0.0
    # calibration on the world the ridge prior does not exactly describe:
    # z^2 near one (pilot 1.00 / 1.02); a broken Hessian gives 0.1 or 10
    z = (Bh - B) / se
    assert 0.6 < np.mean(z ** 2) < 1.6
