"""A top-k depth is a count, so it is an integer.

Every guard truncated first -- `int(k)` in python, `Math.trunc` in the
browser, `as.integer` in R -- and then range-checked the TRUNCATED
value. So `k = 0`, `k = n` and `k > n` were all refused and only a
non-integer slipped through, silently floored:

    top_k_probabilities(mu, 1.5)  ->  the top-1 curve, mass 1
    top_k_probabilities(mu, 2.5)  ->  the top-2 curve, mass 2

The caller asked for a curve that does not exist and got a different
one with no warning -- and the mass they can check is 1 or 2, not the
1.5 or 2.5 they asked about, so nothing downstream reveals it either.
The existing message, "k must be in [1, n-1]", is exactly the sentence
that reads as permitting 1.5.

All three ports shared this, identically; it is a gap in one condition,
not a divergence.
"""
import numpy as np
import pytest

from winning.factor.topk import (loc_scale_from_topk_pair,
                                 top_k_probabilities, top_k_jacobians)

MU = np.array([-0.4, 0.1, 0.2, 0.5])
D = np.array([0.7, 0.8, 0.9, 1.0])


@pytest.mark.parametrize("k", [1, 2, 3, 1.0, 2.0, np.int64(2)])
def test_whole_depths_are_accepted(k):
    """Including a float that IS whole: 2.0 is a depth of two."""
    p = top_k_probabilities(MU, k, D=D)
    assert abs(p.sum() - int(k)) < 1e-9
    assert (p >= 0).all()


@pytest.mark.parametrize("k", [1.5, 2.5, 0.5, 2.0001, 1 + 1e-9])
def test_a_fractional_depth_is_refused(k):
    with pytest.raises(ValueError, match="whole number of places"):
        top_k_probabilities(MU, k, D=D)


@pytest.mark.parametrize("k", [0, 4, 5, -1])
def test_the_range_is_still_enforced(k):
    with pytest.raises(ValueError, match=r"\[1, n-1\]"):
        top_k_probabilities(MU, k, D=D)


def test_the_floored_answer_is_a_real_and_different_curve():
    """Why the silence mattered: 1.5 used to return the top-1 curve,
    which is a perfectly good answer to a question nobody asked. The
    two depths around it give genuinely different numbers, so there was
    nothing odd-looking to notice."""
    one = top_k_probabilities(MU, 1, D=D)
    two = top_k_probabilities(MU, 2, D=D)
    assert np.abs(two - one).max() > 0.1
    assert abs(one.sum() - 1.0) < 1e-9 and abs(two.sum() - 2.0) < 1e-9


def test_every_depth_door_agrees():
    """The contract is on the verb, not on one entry point."""
    for fn in (top_k_probabilities, top_k_jacobians):
        with pytest.raises(ValueError, match="whole number of places"):
            fn(MU, 1.5, D=D)


def test_the_pair_door_checks_both_depths():
    q1 = top_k_probabilities(MU, 1, D=D)
    q2 = top_k_probabilities(MU, 2, D=D)
    loc_scale_from_topk_pair(q1, 1, q2, 2)          # the valid spelling
    with pytest.raises(ValueError, match="k1 must be a whole number"):
        loc_scale_from_topk_pair(q1, 1.5, q2, 2)
    with pytest.raises(ValueError, match="k2 must be a whole number"):
        loc_scale_from_topk_pair(q1, 1, q2, 2.5)
    # and the range and the distinctness are still checked
    with pytest.raises(ValueError, match=r"\[1, n-1\]"):
        loc_scale_from_topk_pair(q1, 0, q2, 2)
    with pytest.raises(ValueError, match="k1 == k2"):
        loc_scale_from_topk_pair(q1, 1, q2, 1)


def test_the_message_says_what_is_wrong():
    with pytest.raises(ValueError) as e:
        top_k_probabilities(MU, 1.5, D=D)
    msg = str(e.value)
    assert "1.5" in msg
    assert "counts finishers" in msg
