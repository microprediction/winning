import numpy as np
from scipy.stats import multivariate_normal

import fastmvn


def test_patch_structured_and_dense_and_unpatch():
    rng = np.random.default_rng(11)
    n = 12
    V = rng.normal(size=(n, 2)) * 0.6
    D = 0.5 + rng.random(n)
    S = V @ V.T + np.diag(D)
    b = rng.normal(size=n) + 1
    baseline = multivariate_normal(mean=np.zeros(n), cov=S).cdf(b)

    fastmvn.patch_scipy()
    try:
        patched = multivariate_normal.cdf(b, mean=np.zeros(n), cov=S)
        assert abs(patched - baseline) < 5e-5
        # frozen-distribution path must also keep working (any route)
        frozen = multivariate_normal(mean=np.zeros(n), cov=S).cdf(b)
        assert abs(frozen - baseline) < 5e-5
        # dense covariance still returns scipy's own answer
        Sd = np.cov(rng.normal(size=(n, 3 * n)))
        dd = multivariate_normal.cdf(b, mean=np.zeros(n), cov=Sd)
        assert np.isfinite(dd)
    finally:
        fastmvn.unpatch_scipy()
    after = multivariate_normal.cdf(b, mean=np.zeros(n), cov=S)
    assert abs(after - baseline) < 5e-5


def test_explicit_api():
    rng = np.random.default_rng(12)
    n = 30
    V = rng.normal(size=(n, 2)) * 0.6
    D = 0.5 + rng.random(n)
    b = rng.normal(size=n) + 1
    p, meth = fastmvn.mvn_cdf_fast_info(upper=b, V=V, D=D)
    ps = multivariate_normal(mean=np.zeros(n),
                             cov=V @ V.T + np.diag(D)).cdf(b)
    assert meth == "factor"
    assert abs(p - ps) < 5e-5
