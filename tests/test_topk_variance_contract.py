"""`D` goes through `as_idio`, in the top-k family too.

`as_idio` is where the idiosyncratic-variance rule is decided -- a scalar
is the same variance for everyone, a wrong length or a negative or
non-finite entry raises -- and the race verbs went through it while the
top-k family called `np.asarray(D, float)` at seven sites.

A scalar therefore became a 0-d array, and the COMPILED kernel indexed
past it:

    pyo3_runtime.PanicException: index out of bounds: the len is 1 but
    the index is 1

A rust panic is not a python exception; it is not something a caller can
catch and recover from in the ordinary way, which makes it worse than
the wrong answer it replaced. On the pure-numpy path the same call gave
`IndexError: invalid index to scalar variable`.

Found while fixing the browser's half of the same contract (#254): the
browser accepted a scalar and python panicked on it, so the ports
disagreed and python was the one that was wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor.topk import rank_probabilities, top_k_probabilities

MU = [-0.4, 0.0, 0.4]


def test_a_scalar_variance_is_the_same_for_every_contestant():
    """This panicked the compiled kernel."""
    a = np.asarray(top_k_probabilities(MU, 2, D=1.0))
    b = np.asarray(top_k_probabilities(MU, 2, D=[1.0, 1.0, 1.0]))
    assert np.abs(a - b).max() < 1e-14
    assert np.isfinite(a).all()


def test_a_scalar_variance_works_for_rank_too():
    a = np.asarray(rank_probabilities(MU, D=0.8))
    b = np.asarray(rank_probabilities(MU, D=[0.8, 0.8, 0.8]))
    assert np.abs(a - b).max() < 1e-14


@pytest.mark.parametrize("D", [[1, 1, 1, 1], [1, 1], [1]])
def test_a_variance_of_the_wrong_length_is_refused(D):
    """The dangerous one: an extra entry used to broadcast into a
    plausible, normalised, materially wrong answer."""
    with pytest.raises(ValueError, match="one idiosyncratic variance"):
        top_k_probabilities(MU, 2, D=D)


@pytest.mark.parametrize("D,match", [
    ([1, -1, 1], "negative"),
    ([1, 0, 1], "strictly positive"),
    ([1, np.nan, 1], "non-finite"),
    ([1, np.inf, 1], "non-finite"),
])
def test_an_impossible_variance_is_refused(D, match):
    with pytest.raises(ValueError, match=match):
        top_k_probabilities(MU, 2, D=D)


def test_the_ordinary_call_is_unchanged():
    p = np.asarray(top_k_probabilities(MU, 2, D=[0.9, 1.0, 1.1]))
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()
    assert abs(p.sum() - 2) < 5e-3          # k slots, within the mass check
