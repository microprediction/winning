"""`as_loadings` checks finiteness, as its two siblings already did.

The three shape contracts are meant to be the one place each rule is
decided, and they should refuse the same kinds of thing. `as_idio`
names a non-finite variance and `as_weights` names a non-finite
weight; `as_loadings` was the one that let a NaN through.

The NaN did not produce a wrong answer -- it travelled to the lattice
sizing and came back as

    ValueError: cannot convert float NaN to integer

which is a refusal, but one that names nothing the caller passed and
points at the wrong module. A caller reading that goes looking in the
lattice code for a bug in their loadings.
"""
import numpy as np
import pytest

from winning.factor.races import race_probabilities
from winning.shapes import as_idio, as_loadings, as_weights

N = 4


@pytest.mark.parametrize("V", [0.5, [1.0, 2.0, 3.0, 4.0],
                               [[1.0, 2.0]] * 4,
                               [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
def test_every_valid_spelling_still_works(V):
    A = as_loadings(V, N)
    assert A.shape[0] == N
    assert np.isfinite(A).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_loading_is_refused(bad):
    with pytest.raises(ValueError, match="non-finite"):
        as_loadings([1.0, bad, 3.0, 4.0], N)


def test_it_is_caught_in_two_dimensions_too():
    with pytest.raises(ValueError, match="non-finite"):
        as_loadings([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]], N)


def test_the_message_names_the_argument_and_counts_them():
    with pytest.raises(ValueError) as e:
        as_loadings([np.nan, 2.0, np.nan, 4.0], N)
    msg = str(e.value)
    assert msg.startswith("V ")
    assert "2 of 4" in msg


def test_the_three_contracts_agree_about_non_finite():
    """The point of the change: one rule, three doors, same answer."""
    for fn, arg in ((as_loadings, ([1.0, np.nan, 3.0, 4.0], N)),
                    (as_idio, ([1.0, np.nan, 1.0, 1.0], N)),
                    (as_weights, ([0.5, np.nan, 1.0],))):
        with pytest.raises(ValueError, match="non-finite"):
            fn(*arg)


def test_the_race_now_refuses_at_the_door():
    """Before, this reached the lattice and failed there."""
    mu = np.array([0.0, 0.3, -0.2, 0.5])
    with pytest.raises(ValueError, match="non-finite"):
        race_probabilities(mu, V=np.array([0.5, np.nan, -0.2, 0.1]),
                           D=np.ones(N))
    # and the clean race is untouched
    p = np.asarray(race_probabilities(mu, V=np.array([0.5, 0.3, -0.2, 0.1]),
                                      D=np.ones(N)))
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9
