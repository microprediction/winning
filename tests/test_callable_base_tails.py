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


@pytest.mark.parametrize("gap", [20.0, 30.0, 35.0])
def test_gumbel_last_place_is_plackett_luce(gap):
    # #710: the min-Gumbel is asymmetric, so its lower tail cannot be
    # the reflected survival, and 1 - S rounded it to zero: last place at
    # gap 35 came back 0 for 2.05e-39 at every resolution.
    c = np.pi / np.sqrt(6.0)
    w0 = np.exp(c * gap)
    exact = 2.0 / ((w0 + 2.0) * (w0 + 1.0))
    mu = [-gap, 0.0, 0.0]
    for points in (513, 8193):
        b = bottom_k_probabilities(mu, 1, base="gumbel", points=points)
        R = rank_probabilities(mu, base="gumbel", points=points)
        assert b[0] == pytest.approx(exact, rel=1e-9)
        assert R[0, -1] == pytest.approx(exact, rel=1e-9)
        assert np.allclose(R.sum(axis=0), 1.0, atol=1e-9)
    # the rare winner of an exacta takes the same lower tail
    w = np.exp(-c * gap)
    o = ordered_probabilities([gap, 0.0, 0.0], 2, base="gumbel")
    assert o[0, 1] == pytest.approx(w / (w + 2.0) / 2.0, rel=1e-9)


def test_gumbel_cdf_matches_its_survival():
    g = BASES["gumbel"]
    z = np.linspace(-6, 6, 25)
    assert np.allclose(_lower_cdf(g)(z) + g(z)[0], 1.0, atol=1e-15)
    for name in ("normal", "logistic", "laplace"):
        assert _lower_cdf(BASES[name]) is None


@pytest.mark.parametrize("base", ["gumbel", "logistic", "laplace"])
def test_ordered_rare_winner_takes_the_stable_lower_tail_598(base):
    # #598: the ordered-prefix path formed every non-normal lower tail
    # as 1 - S; a logistic runner 40 behind won 1e-15 of its true share
    from winning.factor import race_probabilities
    for g in (20.0, 40.0):
        q = ordered_probabilities([g, 0.0, 0.0], 2, base=base, points=2001)
        r = race_probabilities([g, 0.0, 0.0], base=base, points=8193)
        assert q[0].sum() == pytest.approx(r[0], rel=1e-4)


def test_ordered_gumbel_prefix_is_plackett_luce_598():
    from winning.factor.permutations import plackett_luce_prefix_logprob
    for mu in ([0.0, 20.0, 40.0], [0.0, 25.0, 50.0]):
        mu = np.array(mu)
        q = ordered_probabilities(mu, k=3, D=np.full(3, np.pi ** 2 / 6),
                                  base="gumbel", points=1001, mass_tol=1e-5)
        exact = plackett_luce_prefix_logprob(mu, [2, 1, 0])
        assert np.log(q[2, 1, 0]) == pytest.approx(exact, abs=1e-4)
