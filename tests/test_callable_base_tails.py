"""A callable base may carry its own tails: cdf / log_cdf / log_sf (#511
follow-up).

The count program takes a lower-tail CDF from the reflected survival for
the symmetric named bases, keyed on function identity, so a harmless
wrapper of the same law fell back to 1 - S and lost the rare places: the
normal second place of [-20, 0, 0] came out 99% low and [-30, 0, 0] 21
orders low. A wrapper that carries the tails now prices them as the
named base does; one that carries none is unchanged.
"""
import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr

from winning.factor.races import BASES, _lower_cdf
from winning.factor.permutations import ordered_probabilities
from winning.factor.topk import (bottom_k_probabilities, rank_probabilities,
                                 top_k_probabilities)


def _wrap(name, **tails):
    b = BASES[name]

    def w(z):
        return b(z)
    for k, v in tails.items():
        setattr(w, k, v)
    return w


def _tails(name):
    b = BASES[name]
    cdf = ndtr if name == "normal" else (lambda z: b(-np.asarray(z))[0])
    return dict(cdf=cdf, log_sf=b.log_sf)


@pytest.mark.parametrize("name", ["normal", "logistic", "laplace"])
@pytest.mark.parametrize("lead", [-20.0, -30.0])
def test_wrapper_with_tails_matches_the_named_base(name, lead):
    mu = [lead, 0.0, 0.0]
    w = _wrap(name, **_tails(name))
    for f in (lambda b: rank_probabilities(mu, base=b),
              lambda b: bottom_k_probabilities(mu, 1, base=b),
              lambda b: bottom_k_probabilities(mu, 2, base=b),
              lambda b: top_k_probabilities(mu, 2, base=b)):
        ref, got = np.asarray(f(name)), np.asarray(f(w))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_the_plain_wrapper_still_loses_the_tail():
    """The attribute is what does it: without one, 1 - S as before."""
    mu = [-30.0, 0.0, 0.0]
    named = rank_probabilities(mu, base="normal")[0][1]
    plain = rank_probabilities(mu, base=_wrap("normal"))[0][1]
    assert named == pytest.approx(7.212994e-100, rel=1e-6)
    assert plain < 1e-10 * named
    assert _lower_cdf(_wrap("normal")) is None
    assert _lower_cdf(BASES["normal"]) is None


def test_log_cdf_serves_when_cdf_is_absent():
    mu = [-20.0, 0.0, 0.0]
    w = _wrap("normal", log_cdf=log_ndtr)
    np.testing.assert_allclose(rank_probabilities(mu, base=w),
                               rank_probabilities(mu, base="normal"),
                               rtol=1e-12, atol=0)


def test_ordered_prefixes_take_the_cdf():
    mu = [-12.0, 0.0, 0.5, 1.0]
    ref = ordered_probabilities(mu, k=2, base="normal")
    got = ordered_probabilities(mu, k=2, base=_wrap("normal", cdf=ndtr))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=0)


def test_log_sf_prices_an_underflowing_winner():
    mu = [0.0, 53.0, 106.0]
    w = _wrap("normal", **_tails("normal"))
    exact = float(ndtr(-53 / np.sqrt(2)))           # 1.105e-307
    assert top_k_probabilities(mu, 1, base=w)[1] == pytest.approx(exact,
                                                                  rel=1e-6)
    assert rank_probabilities(mu, base=w)[0, 1] == pytest.approx(exact,
                                                                 rel=1e-6)


@pytest.mark.parametrize("tails, err, match", [
    (dict(cdf=0.5), TypeError, "base.cdf must be callable"),
    (dict(cdf=lambda z: ndtr(-z)), ValueError, "non-decreasing"),
    (dict(cdf=lambda z: 1 / (1 + np.exp(-z))), ValueError, "disagrees"),
    (dict(cdf=lambda z: np.full_like(z, np.nan)), ValueError, "finite"),
    (dict(log_sf=lambda z: -log_ndtr(-z)), ValueError, "non-increasing"),
    (dict(log_cdf=lambda z: log_ndtr(z) + 0.1), ValueError, "<= 0"),
    (dict(cdf=lambda z: 0.0), ValueError, "shape"),
])
def test_bad_tails_are_refused(tails, err, match):
    with pytest.raises(err, match=match):
        rank_probabilities([0.0, 1.0, 2.0], base=_wrap("normal", **tails))
