"""Rank counts are not NumPy slice endpoints (#958)."""
import numpy as np
import pytest

from winning.factor.core import fit_covariance

BAD = [-1, -1.0, 0.5, True, np.bool_(False), np.nan, np.inf,
       -np.inf, 1 + 0j, '1', None, [1], np.array([[1]]),
       np.ma.array(1, mask=True)]


@pytest.mark.parametrize('name', ['k', 'm'])
@pytest.mark.parametrize('bad', BAD)
@pytest.mark.parametrize('scale,n', [(1., 1), (1., 3), (16., 3)])
def test_invalid_rank_rejected_before_shortcuts(name, bad, scale, n):
    C = scale * np.eye(n)
    before = C.copy()
    with pytest.raises(ValueError, match=rf'{name}.*non-negative.*integer'):
        fit_covariance(C, **{name: bad}, nodes_log2=3)
    np.testing.assert_array_equal(C, before)


@pytest.mark.parametrize('value', [0, 1, 10])
@pytest.mark.parametrize('convert', [int, np.int64, np.uint64, float,
                                    np.float64, np.asarray])
def test_valid_scalar_representations_match_integer(value, convert):
    C = np.array([[1., .2, -.1], [.2, 1.5, .3], [-.1, .3, 2.]])
    expected = fit_covariance(C, k=value, m=value, nodes_log2=3)
    actual = fit_covariance(C, k=convert(value), m=convert(value), nodes_log2=3)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_large_integer_rank_still_clips_to_field_size():
    C = np.eye(3)
    expected = fit_covariance(C, k=3, m=3, nodes_log2=3)
    actual = fit_covariance(C, k=10**100, m=10**100, nodes_log2=3)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["k", "m"])
@pytest.mark.parametrize("value", [np.timedelta64(1, "ns"), np.timedelta64(1, "D"),
                                   np.datetime64("2026-01-01")])
def test_temporal_scalars_are_not_counts(name, value):
    # numpy registers timedelta64 as Integral: 1 ns became rank 1 and 1 D
    # leaked a TypeError, so the contract depended on the time unit
    kw = dict(k=1, m=0)
    kw[name] = value
    with pytest.raises(ValueError, match=rf'{name}.*non-negative.*integer'):
        fit_covariance(np.eye(3), nodes_log2=3, **kw)


@pytest.mark.parametrize("name", ["k", "m"])
def test_huge_ranks_clip_and_huge_negatives_refuse_without_formatting(name):
    big = 10 ** 5000          # past Python's default int-to-str digit limit
    kw = dict(k=1, m=0)
    kw[name] = big
    V, D, F, W = fit_covariance(np.eye(3), nodes_log2=3, **kw)
    assert np.isfinite(D).all()
    kw[name] = -big
    with pytest.raises(ValueError, match=rf'{name}.*non-negative.*integer'):
        fit_covariance(np.eye(3), nodes_log2=3, **kw)
