"""The rank matrix that is RETURNED satisfies the identities it claims.

`rank_probabilities` checked the raw quadrature matrix, then divided by the
raw row sums and returned that. Row normalisation is not neutral: it makes
every row exact and MOVES the columns, so a matrix that passed the check
could fail the stated identity afterwards and nothing looked. On a field
whose sds span 73x it pushed the column excess from 4.5e-3 to 5.4e-3, and
the documented identity -- cumulative rank rows reproduce
top_k_probabilities -- broke by 2.6e-3 (#203).

The columns are deliberately NOT forced to sum to one. Alternating
row/column scaling would make both identities exact, and would do it by
hiding the under-resolution that caused the defect: the same field at 2001
points has a column error of 3.5e-9, so the quadrature is what is wrong,
not the normalisation.
"""
import numpy as np
import pytest

from winning.factor.topk import rank_probabilities, top_k_probabilities

# the reported field: sds from 0.19 to 14.3, a 73x spread
MU = np.array([0.852479112355, 1.342705472309, -4.927823648082])
SD = np.array([0.635892136598, 0.194587863958, 14.30720080209])


def test_an_underresolved_field_raises_instead_of_returning_a_bad_matrix():
    with pytest.raises(RuntimeError, match="RETURNED matrix"):
        rank_probabilities(MU, D=SD ** 2, points=513)


@pytest.mark.parametrize("points", [2001, 8001])
def test_both_identities_hold_on_the_returned_matrix(points):
    P = rank_probabilities(MU, D=SD ** 2, points=points)
    assert np.abs(P.sum(axis=1) - 1).max() < 5e-3      # a runner takes one rank
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3      # a rank is taken once
    assert np.isfinite(P).all()
    assert (P >= 0).all() and (P <= 1).all()


@pytest.mark.parametrize("k", [1, 2])
def test_cumulative_rows_reproduce_top_k(k):
    """The documented identity, which row normalisation had broken by
    2.6e-3 on this field: it now holds to the quadrature's own error."""
    P = rank_probabilities(MU, D=SD ** 2, points=2001)
    q = np.asarray(top_k_probabilities(MU, k, D=SD ** 2, points=2001))
    assert np.abs(P[:, :k].sum(axis=1) - q).max() < 1e-6


def test_an_ordinary_field_is_untouched():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=6)
    D = 0.5 + rng.random(6)
    P = rank_probabilities(mu, D=D, points=513)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-9
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3
    q = np.asarray(top_k_probabilities(mu, 2, D=D, points=513))
    assert np.abs(P[:, :2].sum(axis=1) - q).max() < 1e-6
