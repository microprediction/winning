"""Every observation is accounted for, or the likelihood refuses.

The exact-likelihood cores in all three ports iterate over the LEGAL
alternative labels and gather the rows matching each, so a row whose
choice is outside the range is never visited. It contributed nothing to
the log-likelihood and a zero row to the score, and the fit silently
optimised a SUBSET of the data while reporting it as the whole.

The direction is what makes it dangerous. Dropping observations RAISES
the log-likelihood, because there is less of it: five bad rows out of
forty moved it from -55.699 to -49.134. Nothing downstream can tell that
from a better fit.

Julia additionally accepted a choice vector SHORTER than T and dropped
the tail; R left `logp == 0` for NA rows, which adds zero negative
log-likelihood and zero gradient (#194).
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.likelihood import choice_loglik_and_score

T, J, R = 40, 3, 1


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(T, J)),
            rng.normal(size=(J, R)) * 0.4,
            rng.integers(0, J, size=T))


def test_a_valid_panel_still_fits():
    mu, V, choice = _fixture()
    ll, dmu, dV = choice_loglik_and_score(mu, V, choice)
    assert np.isfinite(ll)
    assert dmu.shape == (T, J) and dV.shape == (J, R)


def test_dropping_observations_would_raise_the_log_likelihood():
    """The property that makes silence dangerous, asserted rather than
    described: fewer observations is a BETTER number."""
    mu, V, choice = _fixture()
    full = choice_loglik_and_score(mu, V, choice)[0]
    part = choice_loglik_and_score(mu[5:], V, choice[5:])[0]
    assert part > full


@pytest.mark.parametrize("bad_value", [J, J + 96, -1])
def test_a_choice_outside_the_alternatives_is_refused(bad_value):
    mu, V, choice = _fixture()
    choice = choice.copy()
    choice[:5] = bad_value
    with pytest.raises(ValueError, match=r"outside 0\.\.2"):
        choice_loglik_and_score(mu, V, choice)


def test_the_message_counts_how_many_and_names_the_first():
    mu, V, choice = _fixture()
    choice = choice.copy()
    choice[3] = 99
    with pytest.raises(ValueError, match=r"choice\[3\] = 99.*1 of 40"):
        choice_loglik_and_score(mu, V, choice)


@pytest.mark.parametrize("n", [T - 10, T + 1])
def test_a_choice_vector_of_the_wrong_length_is_refused(n):
    mu, V, choice = _fixture()
    with pytest.raises(ValueError, match="one entry per observation"):
        choice_loglik_and_score(mu, V, np.resize(choice, n))


def test_a_missing_choice_is_refused_rather_than_scored_zero():
    mu, V, choice = _fixture()
    ch = choice.astype(float)
    ch[:2] = np.nan
    with pytest.raises(ValueError, match="missing"):
        choice_loglik_and_score(mu, V, ch)


def test_a_non_integer_choice_is_refused():
    mu, V, choice = _fixture()
    with pytest.raises(ValueError, match="integer"):
        choice_loglik_and_score(mu, V, choice.astype(float) + 0.5)


def test_an_integer_valued_float_is_accepted():
    """Refusing 1.0 would be pedantry, not a guard."""
    mu, V, choice = _fixture()
    a = choice_loglik_and_score(mu, V, choice)[0]
    b = choice_loglik_and_score(mu, V, choice.astype(float))[0]
    assert abs(a - b) < 1e-12
