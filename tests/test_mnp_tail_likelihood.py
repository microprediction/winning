"""The MNP tail likelihood is finite and its score is not zero.

#270 reports three compounding defects. This closes the two that make
an optimizer lie to you, and deliberately not the third.

`logPhi` was `log(max(ndtr(a), 1e-300))`. `ndtr` underflows to exactly
zero below about -37, so every deep tail collapsed to the SAME value,
log(1e-300) = -690.78: the objective went FLAT. The Mills ratio, taken
as `exp(logphi - logPhi)`, then underflowed to a score of exactly
ZERO. Together those mean an optimizer declares convergence precisely
where an observation is most badly contradicted -- it cannot tell a
maximum from a floor.

`log_ndtr` removes both. What it does NOT remove is the quadrature
error underneath: the fixed Gauss-Hermite rule samples the chosen
alternative's own noise where the PRIOR has mass, and for a large
observed contrast the integrand's mass is far outside it -- at a gap of
-20 the mode is at z = 10 and a 7-node rule reaches |z| < 3.8. So the
values here are still wrong, by 38% at a gap of -20, and this file
PINS that rather than pretending otherwise. A Laplace-tilted rule fixes
it to machine precision, but the tilt makes the quadrature depend on
the parameters, which breaks the identity between the analytic score
and the gradient of the computed objective and left `fit` reporting
`converged = False` after ten times the work. That needs the tilt's own
derivative, and belongs in its own change.
"""
import numpy as np
import pytest
from scipy.special import log_ndtr

from winning.likelihood import choice_loglik_and_score

DEEP = [-40.0, -60.0, -100.0, -200.0]


def _oracle(gap):
    x = gap / np.sqrt(2.0)
    ll = log_ndtr(x)
    return ll, np.exp(-0.5 * x * x - 0.5 * np.log(2.0 * np.pi) - ll) / np.sqrt(2.0)


def _binary(gap, **kw):
    return choice_loglik_and_score(np.array([[gap, 0.0]]), np.zeros((2, 1)),
                                   np.array([0]), **kw)


def test_the_objective_does_not_go_flat():
    """The floor made every gap past about -37 return -690.78, so the
    objective was constant exactly where it should be steepest."""
    lls = [_binary(g)[0] for g in DEEP]
    assert all(np.isfinite(lls)), lls
    for a, b in zip(lls, lls[1:]):
        assert b < a - 1.0, (a, b)          # strictly falling, and by a lot
    assert min(lls) < -1e4                  # far past the old -690.78 floor


def test_the_score_is_not_zero_and_points_the_right_way():
    """A zero gradient is indistinguishable from a converged optimum.
    The magnitude is still wrong -- see the module docstring -- but the
    sign and the direction of change are right, so an optimizer is
    pushed towards the truth rather than stopped dead."""
    prev = None
    for gap in DEEP:
        _, dmu, _ = _binary(gap)
        _, exact = _oracle(gap)
        assert dmu[0, 0] > 0.0, (gap, dmu[0, 0])
        assert np.isfinite(dmu).all()
        # d logP / d mu_0 grows as the observation gets more surprising
        if prev is not None:
            assert dmu[0, 0] > prev
        prev = dmu[0, 0]
        # and what one alternative gains the other loses
        assert abs(dmu[0, 0] + dmu[0, 1]) < 1e-9 * dmu[0, 0]


def test_a_central_difference_is_not_zero_either():
    """The report's own check: the analytic score was differentiating a
    floored objective, and the objective itself had gone flat, so a
    central difference of it was exactly zero too."""
    h = 1e-4
    for gap in (-60.0, -100.0):
        fd = (_binary(gap + h)[0] - _binary(gap - h)[0]) / (2 * h)
        assert fd > 0.5, (gap, fd)
        _, dmu, _ = _binary(gap)
        # the analytic score is still the exact derivative of what is
        # computed, which is the property the fix must not break
        assert abs(fd - dmu[0, 0]) < 1e-4 * dmu[0, 0], (gap, fd, dmu[0, 0])


def test_moderate_gaps_are_accurate():
    """Where the rule does reach the integrand, the answer is right --
    so the tail failure is the quadrature, not the formula."""
    for gap in (-1.0, -2.0, -3.0):
        ll, dmu, _ = _binary(gap)
        ex, es = _oracle(gap)
        assert abs(ll - ex) / abs(ex) < 1e-3, (gap, ll, ex)
        assert abs(dmu[0, 0] - es) / es < 1e-2, (gap, dmu[0, 0], es)


@pytest.mark.parametrize("gap", [-20.0, -60.0])
def test_the_remaining_quadrature_error_is_recorded(gap):
    """Not a pass mark: a measurement, so that a later change closing
    #270's third defect has something to move. At a gap of -20 the
    likelihood is 38% away from the closed form and the score 62%."""
    ll, dmu, _ = _binary(gap)
    ex, es = _oracle(gap)
    assert abs(ll - ex) / abs(ex) > 0.3
    assert abs(dmu[0, 0] - es) / es > 0.5
    # but bounded, and on the correct side: the rule UNDERSTATES the
    # probability, so the log-likelihood is too negative
    assert ll < ex
    assert 1.0 < abs(ll / ex) < 3.0


def test_an_extreme_observation_does_not_poison_the_ordinary_ones():
    mu = np.array([[0.2, 0.0, -0.1], [-60.0, 0.0, 0.0], [0.5, 0.3, 0.1]])
    V = np.zeros((3, 1))
    ch = np.array([0, 0, 1])
    ll, dmu, _ = choice_loglik_and_score(mu, V, ch)
    assert np.isfinite(ll) and np.isfinite(dmu).all()
    alone = choice_loglik_and_score(mu[[0, 2]], V, ch[[0, 2]])
    assert np.abs(dmu[[0, 2]] - alone[1]).max() < 1e-12


def test_log_ndtr_is_where_the_floor_was():
    """Guard the specific mechanism: the conditional log-CDF must not
    saturate. -690.78 is log(1e-300), the old floor."""
    from winning.likelihood import _log_mills
    a = np.array([-10.0, -37.0, -40.0, -100.0, -300.0])
    lp = log_ndtr(a)
    assert (np.diff(lp) < 0).all()                  # strictly decreasing
    assert lp[-1] < -40000.0                        # nowhere near a floor
    lam = np.exp(_log_mills(a))
    assert (lam > 0).all() and np.isfinite(lam).all()
    assert (np.diff(lam) > 0).all()                 # Mills ratio grows as ~|a|
