"""fastmvn.patch_scipy() must keep scipy's contract: it may answer
faster, never differently, and never accept a call scipy refuses or
refuse one scipy accepts."""
import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDALONE = os.path.join(ROOT, "python", "fastmvn", "src")


def _load(name):
    if not os.path.isdir(STANDALONE):
        pytest.skip("standalone fastmvn source not in this tree")
    if STANDALONE not in sys.path:
        sys.path.insert(0, STANDALONE)
    return importlib.import_module(name)


# --- #332 / #333: the scipy patch keeps scipy's contract ----------------

@pytest.fixture
def patched():
    fastmvn = _load("fastmvn")
    fastmvn.patch_scipy()
    try:
        yield fastmvn
    finally:
        fastmvn.unpatch_scipy()


def test_patch_honours_allow_singular_false(patched):
    from scipy.stats import multivariate_normal
    v = np.array([1.0, 2.0, 3.0])
    S = np.outer(v, v)
    with pytest.raises(np.linalg.LinAlgError):
        multivariate_normal.cdf(np.zeros(3), cov=S, allow_singular=False)
    assert multivariate_normal.cdf(np.zeros(3), cov=S,
                                   allow_singular=True) == \
        pytest.approx(0.5, abs=1e-6)
    # a positive-definite factor covariance still takes the fast path
    Sp = S + np.eye(3)
    assert multivariate_normal.cdf(np.zeros(3), cov=Sp) == pytest.approx(
        patched.mvn_cdf_fast(upper=np.zeros(3), sigma=Sp), abs=1e-15)


def test_patch_accepts_scipys_rng(patched):
    import inspect
    from scipy.stats import multivariate_normal
    from scipy.stats._multivariate import multivariate_normal_gen
    orig = patched.patch._ORIGINAL["cdf"]
    if "rng" not in inspect.signature(orig).parameters:
        pytest.skip("this scipy predates rng= on multivariate_normal.cdf")
    rng = np.random.default_rng(123)
    # structured (fast path ignores rng) and dense (forwarded) both bind
    assert multivariate_normal.cdf(np.zeros(3), cov=np.eye(3), rng=rng) == \
        pytest.approx(0.125, abs=1e-12)
    Sd = np.array([[1.0, 0.138873, 0.743785, -0.738739],
                   [0.138873, 1.0, 0.579169, 0.506125],
                   [0.743785, 0.579169, 1.0, -0.37101],
                   [-0.738739, 0.506125, -0.37101, 1.0]])
    assert patched.factorize_covariance(Sd) is None     # scipy's route
    got = multivariate_normal.cdf(np.zeros(4), cov=Sd,
                                  rng=np.random.default_rng(5))
    want = orig(multivariate_normal_gen(), np.zeros(4), cov=Sd,
                rng=np.random.default_rng(5))
    assert got == want
    with pytest.raises(TypeError):
        multivariate_normal.cdf(np.zeros(3), cov=np.eye(3), bogus=1)
