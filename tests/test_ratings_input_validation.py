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
