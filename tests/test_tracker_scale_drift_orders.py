"""AbilityTracker: scale covariance, zero elapsed time, partial orders under
the augmented sampler, and tune_block_rho grid edges by value."""
import numpy as np
import pytest

from winning.ratings.tracker import AbilityTracker, tune_block_rho


def _walk(c):
    # one Gaussian model in ability unit c: means x c, variances x c^2
    trk = AbilityTracker(drift=0.02 * c * c, init_var=4.0 * c * c, beta2=1.0 * c * c)
    rng = np.random.default_rng(7)
    for t in range(12):
        trk.observe(["a", "b", "c"], float(t), scores=list(c * rng.normal(size=3)))
    return trk


@pytest.mark.parametrize("c", [1e-4, 1e-6, 1e-9])
def test_the_tracker_is_scale_covariant(c):
    # the stored variance had an absolute 1e-9 floor: at c = 1e-6 the win
    # probabilities were 0.26 off the unit-1 answer
    one, small = _walk(1.0), _walk(c)
    for k in "abc":
        assert small.state[k].mean / c == pytest.approx(one.state[k].mean, rel=1e-10, abs=1e-12)
        assert small.state[k].var / c ** 2 == pytest.approx(one.state[k].var, rel=1e-10)
    np.testing.assert_allclose(small.predict(["a", "b", "c"], 12.0),
                               one.predict(["a", "b", "c"], 12.0), atol=1e-12)


def test_no_elapsed_time_adds_no_drift():
    # 0.0 ** 0 is 1: drift_exp = 0 added the full drift at dt = 0
    for de in (0.0, 0.5):
        trk = AbilityTracker(drift=0.5, drift_exp=de)
        trk.observe(["a", "b"], 1.0, scores=[0.0, 1.0])
        v = trk.state["a"].var
        p0 = trk.predict(["a", "b"], 1.0)
        trk2 = AbilityTracker(drift=0.5, drift_exp=0.5)
        trk2.observe(["a", "b"], 1.0, scores=[0.0, 1.0])
        np.testing.assert_allclose(p0, trk2.predict(["a", "b"], 1.0), atol=1e-15)
        assert trk.state["a"].var == v
    with pytest.raises(ValueError, match="drift_exp"):
        AbilityTracker(drift_exp=-0.5)


@pytest.mark.parametrize("order", [[2, 0], [1]])
def test_augmented_reads_a_partial_order_as_exact_does(order):
    # the Gibbs sampler indexed a full-length latent vector by the partial
    # order and raised; unranked runners keep their prior, as under exact
    prior = dict(a=(0.5, 1.0), b=(-0.3, 2.0), c=(0.0, 0.5), d=(0.2, 0.8))
    out = {}
    for method in ("exact", "augmented"):
        trk = AbilityTracker(order_method=method, n_aug=4000, burn=200, seed=3)
        ids = list(prior)
        from winning.ratings.tracker import AbilityState
        for k, (m, v) in prior.items():
            trk.state[k] = AbilityState(m, v, 0.0)
        trk.observe(ids, 0.0, order=order)
        out[method] = {k: (trk.state[k].mean, trk.state[k].var) for k in ids}
    ranked = {"abcd"[i] for i in order} if len(order) > 1 else set()
    for k in "abcd":
        if k not in ranked:
            assert out["augmented"][k] == prior[k] == out["exact"][k]
        else:
            assert out["augmented"][k][0] == pytest.approx(out["exact"][k][0], abs=0.05)


def test_tune_block_rho_judges_edges_by_value():
    rng = np.random.default_rng(0)
    contests = []
    for t in range(40):
        runners = list(rng.choice(12, size=4, replace=False))
        contests.append({"runners": [str(r) for r in runners], "t": float(t),
                         "scores": list(rng.normal(size=4)), "groups": [r % 2 for r in runners]})
    a = tune_block_rho(contests, rho_grid=(0.0, 0.2, 0.4))
    b = tune_block_rho(contests, rho_grid=(0.2, 0.0, 0.4))
    assert a["rho"] == b["rho"] and a["at_edge"] == b["at_edge"]
    assert b["at_edge"] == (b["rho"] in (0.0, 0.4))
