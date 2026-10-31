"""The zero-dimensional Gaussian has probability one (#905).

A rectangle in zero dimensions holds its one empty tuple, so its
probability is the empty product, 1, which the R and Julia ports return.
Both python spellings raised "zero-size array to reduction operation
maximum" in the extreme-row rescaler. This is the field dimension n = 0,
not #68's rank-zero loading V of shape (n, 0) at n > 0.
"""
import numpy as np
import pytest

from winning.fastmvn import _rescale_extreme_rows, mvn_cdf_fast, mvn_cdf_fast_info


@pytest.mark.parametrize("kw", [dict(sigma=np.empty((0, 0))),
                                dict(V=np.empty((0, 0)), D=np.empty(0)),
                                dict(V=np.empty((0, 3)), D=np.empty(0)),
                                dict(sigma=np.empty((0, 0)), lower=np.empty(0), upper=np.empty(0), mean=np.empty(0))])
def test_zero_dimensional_gaussian_is_the_empty_product(kw):
    assert mvn_cdf_fast(**kw) == 1.0
    assert mvn_cdf_fast_info(**kw) == (1.0, "factor")


def test_rank_zero_at_positive_dimension_is_unchanged_68():
    # the independent product, not the empty one
    p = mvn_cdf_fast(V=np.empty((2, 0)), D=np.ones(2), upper=np.zeros(2))
    assert abs(p - 0.25) < 1e-12


def test_rescaler_is_the_identity_on_zero_rows():
    e = np.empty(0)
    out = _rescale_extreme_rows(np.empty((0, 2)), e, e, e, e)
    assert all(a.shape[0] == 0 for a in out)


def test_standalone_copy_is_in_lockstep():
    import os
    import fastmvn.core as standalone
    import winning.fastmvn as vendored
    with open(standalone.__file__, "rb") as a, open(vendored.__file__, "rb") as b:
        assert a.read() == b.read()
