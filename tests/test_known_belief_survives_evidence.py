"""Zero belief variance means perfectly known, and evidence must respect it.

`as_variance` documents zero as "a perfectly known quantity" and both
updates accepted it. `update_market` then divided by it, and ONE known
coordinate returned NaN for every runner. `update_winner`'s means were
already right -- `m + v * g` cannot move a coordinate with v = 0 -- but
its variance floor handed that coordinate 1e-6 of uncertainty it had
just been told did not exist (#78).

The limit is not singular. u = 1/(1/v + 1/tau2) -> 0 and
b = m/v + y/tau2 -> inf, but u * b -> m exactly; it was evaluated as
0 * inf. The closed forms agree with the naive ones to every digit
where v > 0, which the last test here pins.
"""
import numpy as np
import pytest

from winning.ratings.market import update_market
from winning.ratings.nway import update_winner

M = np.array([0.0, 0.2, -0.1])
P = np.array([0.5, 0.3, 0.2])


@pytest.mark.parametrize("tau2", [0.25, np.array([0.2, 0.4, 0.3])])
def test_a_fully_known_belief_is_unmoved_by_a_market(tau2):
    mu, var, _ = update_market(M, np.zeros(3), P, tau2=tau2)
    assert np.allclose(mu, M)
    assert np.allclose(var, 0.0)


@pytest.mark.parametrize("tau2", [0.25, np.array([0.2, 0.4, 0.3])])
def test_one_known_coordinate_does_not_poison_the_others(tau2):
    """The reported symptom: a single zero returned NaN for every
    runner, not just for the coordinate that was zero."""
    v = np.array([0.0, 1.0, 1.0])
    mu, var, _ = update_market(M, v, P, tau2=tau2)
    assert np.isfinite(mu).all() and np.isfinite(var).all()
    assert mu[0] == M[0] and var[0] == 0.0
    # and the free coordinates actually moved, so this is an update and
    # not a no-op that happens to be finite
    assert not np.allclose(mu[1:], M[1:])
    assert (var[1:] > 0).all()


def test_a_known_belief_stays_known_through_a_winner():
    for v in (np.zeros(3), np.array([0.0, 1.0, 1.0])):
        mu, var, _ = update_winner(M, v, 0)
        known = v <= 0
        assert np.allclose(mu[known], M[known])
        assert np.allclose(var[known], 0.0), "uncertainty was manufactured"


def test_the_floor_still_applies_where_the_variance_is_positive():
    """The floor exists so a downstream 1/v is finite; only the declared
    -known coordinates are exempt."""
    mu, var, _ = update_winner(M, np.full(3, 1e-12), 0)
    assert (var >= 1e-6).all()
    mu2, var2, _ = update_market(M, np.full(3, 1e-12), P)
    assert (var2 >= 1e-6).all()


def test_zero_is_the_limit_of_small_not_a_special_case():
    """If the closed form were merely a special case bolted on, it could
    disagree with the naive path just above zero. It is the same
    function, so the sequence has to converge to it."""
    exact_mu, exact_var, _ = update_market(M, np.zeros(3), P)
    prev = None
    for v0 in (1e-2, 1e-4, 1e-6, 1e-8):
        mu, var, _ = update_market(M, np.full(3, v0), P)
        gap = np.abs(mu - exact_mu).max()
        assert prev is None or gap <= prev + 1e-12, "not converging"
        prev = gap
    assert prev < 1e-7, f"limit not approached: {prev:.2e}"


def test_positive_variances_are_untouched_by_this_change():
    """Pinned values, READ OFF THE PRE-FIX CODE, so a future rewrite of
    the closed forms cannot quietly move the ordinary path. (My first
    draft of this test carried invented numbers and failed; a pin whose
    values were guessed pins nothing.)"""
    mu, var, logz = update_market(M, np.array([0.5, 1.0, 2.0]), P)
    assert mu.sum() == pytest.approx(-0.2412599493, abs=1e-9)
    assert var.sum() == pytest.approx(1.3160919540, abs=1e-9)
    assert logz == pytest.approx(-2.2584369271, abs=1e-9)
