"""fit_factor_model_missing refuses non-count controls (#873).

int(True) == True, so k=True fitted a one-factor model and k=False the
independent one, both reported converged; min_pairs, max_iter, n_starts
and k_max took Booleans the same way.
"""
import numpy as np
import pytest

from winning.factor import fit_factor_model_missing

rng = np.random.default_rng(6681)
F_ = rng.normal(size=(80, 1))
Y = F_ @ np.array([[1.2], [-0.8], [0.5], [1.0]]).T + rng.normal(scale=0.35, size=(80, 4))
Y[rng.random(Y.shape) < 0.12] = np.nan

BAD = [True, False, np.bool_(True), np.bool_(False), 1.5, np.nan, np.inf, -np.inf, "1", [1]]


@pytest.mark.parametrize("value", BAD)
@pytest.mark.parametrize("name", ["k", "k_max", "min_pairs", "max_iter", "n_starts"])
def test_non_counts_are_refused(name, value):
    kw = {"k": 1} if name != "k_max" else {"k": None}
    kw[name] = value
    with pytest.raises(ValueError, match=name):
        fit_factor_model_missing(Y, **kw)


@pytest.mark.parametrize("name, value", [("k", -1), ("k_max", -1), ("min_pairs", -1),
                                         ("max_iter", -1), ("n_starts", 0)])
def test_below_the_minimum_is_refused(name, value):
    kw = {"k": 1} if name != "k_max" else {"k": None}
    kw[name] = value
    with pytest.raises(ValueError, match=name):
        fit_factor_model_missing(Y, **kw)


@pytest.mark.parametrize("k", [1, np.int64(1), 1.0, np.float64(1.0)])
def test_whole_numbers_in_any_spelling_are_the_same_fit(k):
    V, D, rep = fit_factor_model_missing(Y, k=k, max_iter=np.int32(200), n_starts=np.int64(1),
                                         min_pairs=np.uint8(1), tol=1e-9)
    V1, D1, rep1 = fit_factor_model_missing(Y, k=1, max_iter=200, tol=1e-9)
    assert rep["k"] == 1
    np.testing.assert_array_equal(V, V1)
    np.testing.assert_array_equal(D, D1)
