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
