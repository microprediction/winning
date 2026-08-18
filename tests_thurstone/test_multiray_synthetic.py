import numpy as np

from winning.thurstone import AbilityCalibrator, MultiRayGlobalCalibrator


def test_multiray_probability_fit(base):
    rng = np.random.default_rng(12345)
    dim = 2
    n_items = 25
    n_conds = 6

    item_ids = [f"h{i}" for i in range(n_items)]

    # True embeddings and condition rays
    Z_true = rng.standard_normal((n_items, dim))
    V_true = []
    beta_true = []
    for _ in range(n_conds):
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v)
        V_true.append(v)
        beta_true.append(0.1 * rng.standard_normal())
    V_true = np.array(V_true, dtype=float)
    beta_true = np.array(beta_true, dtype=float)

    # Observations per condition
    calibrators = []
    prices_obs = []
    cond_ids = []
    for j in range(n_conds):
        a = beta_true[j] + Z_true @ V_true[j]
        # Build densities for state pricing
        dens = [base.shift_fractional(float(ai / base.lattice.unit)) for ai in a]
        from winning.thurstone.pricing import Race

        p_obs = np.array(Race(dens).state_prices(), dtype=float)
        cal_j = AbilityCalibrator(base)
        calibrators.append(cal_j)
        prices_obs.append(p_obs)
        cond_ids.append(f"c{j}")

    # Fit model
    fit = MultiRayGlobalCalibrator(item_ids=item_ids, dim=dim, random_state=999)
    for j in range(n_conds):
        fit.add_condition(
            cond_id=cond_ids[j],
            calibrator=calibrators[j],
            item_ids=item_ids,
            prices=prices_obs[j],
        )

    # Before fitting
    fit.rebuild_all_curves()
    preds0 = []
    for j in range(n_conds):
        preds0.append(fit.predict_condition(cond_ids[j]))
    preds0 = np.stack(preds0, axis=0)
    obs = np.stack(prices_obs, axis=0)
    mse0 = float(np.mean((preds0 - obs) ** 2))

    # Fit
    fit.fit_with_rebuild(num_outer_iters=5, num_inner_iters=10)

    # After fitting
    fit.rebuild_all_curves()
    preds1 = []
    for j in range(n_conds):
        preds1.append(fit.predict_condition(cond_ids[j]))
    preds1 = np.stack(preds1, axis=0)
    mse1 = float(np.mean((preds1 - obs) ** 2))

    # Quality checks
    assert mse1 < mse0
    assert mse1 < 1e-3
    max_abs_err = float(np.max(np.abs(preds1 - obs)))
    assert max_abs_err < 5e-2
