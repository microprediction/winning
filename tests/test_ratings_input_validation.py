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
