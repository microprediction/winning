"""A one-runner field has no covariance structure, and must not crash.

`race_probabilities([mu])` already returned `[1]`. Supplying the
equivalent `cov=[[v]]` crashed: the inner solver divides by
`(1 - 2/n) + n/n^2`, which is `-1 + 1 = 0` at `n = 1`. The `n <= 2`
branch added for #181 divides by the same quantity, so it covered `n = 2`
and left `n = 1` exactly where it was (#273).

R failed at the same field for a different reason -- it asks
`.factor_model_projected` for `min(k, n - 1) = 0` factors and gets a
`0 x 0` matrix back, which then fails to multiply.

There is nothing for a factor to correlate with one runner, so the whole
variance is idiosyncratic and the fit is exact at rank zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor import race_probabilities
from winning.factor.core import fit_covariance


def test_the_one_runner_race_is_certain_with_a_covariance():
    p = np.asarray(race_probabilities(np.array([2.0]),
                                      cov=np.array([[4.0]])))
    assert p.shape == (1,)
    assert abs(p[0] - 1.0) < 1e-12


def test_it_agrees_with_the_plain_one_runner_race():
    a = np.asarray(race_probabilities(np.array([2.0])))
    b = np.asarray(race_probabilities(np.array([2.0]), cov=np.array([[4.0]])))
    assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("v", [1e-6, 1.0, 4.0, 1e6])
def test_any_positive_variance_works(v):
    V, D, F, W = fit_covariance(np.array([[v]]))
    assert V.shape[0] == 1
    assert D.shape == (1,)
    assert abs(D[0] - v) < 1e-9 * max(v, 1.0)
    assert abs(float(np.sum(W)) - 1.0) < 1e-12


def test_the_report_has_the_keys_every_caller_reads():
    """`_fit_cov` reads rank, projected_residual_max, sharpness and
    contrast_residual_max. A short report raises KeyError several frames
    away from the cause."""
    *_, report = fit_covariance(np.array([[4.0]]), return_report=True)
    for key in ("rank", "projected_residual_max", "sharpness",
                "contrast_residual_max", "projected_residual_rel"):
        assert key in report, key
    assert report["rank"] == 0          # nothing for a factor to explain
    assert report["projected_residual_max"] == 0.0


@pytest.mark.parametrize("n", [2, 3, 5])
def test_larger_fields_are_untouched(n):
    C = np.eye(n) + 0.2 * (np.ones((n, n)) - np.eye(n))
    mu = np.linspace(0.0, 1.0, n)
    p = np.asarray(race_probabilities(mu, cov=C))
    assert p.shape == (n,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()


# --- the validation the one-runner branch must sit BELOW (#277) -------
#
# The early return keys off nrow alone, so it has to come after the
# shape and covariance checks or a 1 x 2 matrix -- nrow 1 -- gets
# answered from C[1, 1] with the second column dropped. R had no
# validation at all before this; python's squareness was only ever
# caught by accident, because a 1 x 2 broadcasts against its transpose
# into a 2 x 2 and reported "not symmetric" instead.

@pytest.mark.parametrize("C, message", [
    ([[1.0, 2.0]], "must be square"),          # nrow 1, ncol 2
    ([[1.0], [2.0]], "must be square"),        # nrow 2, ncol 1
    ([1.0, 2.0, 3.0], "must be square"),       # 1-D: np.diag would BUILD a matrix
    ([[[1.0]]], "must be square"),             # 3-D
])
def test_a_non_square_cov_is_refused(C, message):
    with pytest.raises(ValueError, match=message):
        fit_covariance(np.array(C))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_one_runner_cov_is_refused(bad):
    with pytest.raises(ValueError, match="NaN or inf"):
        fit_covariance(np.array([[bad]]))


def test_a_negative_one_runner_variance_is_refused():
    """Not clamped to the floor: a negative variance is not a
    covariance, and answering [1] for it would dress invalid data as a
    certain race."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_covariance(np.array([[-1.0]]))


def test_a_zero_one_runner_variance_is_accepted_at_the_floor():
    """Zero IS positive semidefinite, so it is valid input; it lands on
    the same 1e-12 floor the R port uses."""
    V, D, F, W = fit_covariance(np.array([[0.0]]))
    assert float(D[0]) == 1e-12
    assert float(W[0]) == 1.0


def test_the_front_door_refuses_them_too():
    """race_probabilities(mu, cov=) calls the fitter directly, so these
    must not be reachable through it either."""
    for C, message in [([[-1.0]], "positive semidefinite"),
                       ([[np.nan]], "NaN or inf"),
                       ([[1.0, 2.0]], "must be square")]:
        with pytest.raises(ValueError, match=message):
            race_probabilities([0.0], cov=C)


# --- symmetrising must not overflow what the check just passed (#279)
#
# The finiteness check runs BEFORE the symmetrisation, and
# `0.5 * (C + C.T)` doubles before it halves. So a finite variance near
# the double ceiling passed the check and then became inf, and the inf
# came back out as D -- the front door raised "D has a non-finite
# entry" for input it had already accepted as finite. Halving each term
# first is the same number everywhere else: over 2000 random matrices
# spanning 400 orders of magnitude the two spellings are bit-identical,
# because multiplying by 0.5 is exact.

@pytest.mark.parametrize("v", [1e307, 1e308, 1.5e308,
                               float(np.finfo(float).max)])
def test_a_huge_finite_variance_survives_symmetrising(v):
    V, D, F, W = fit_covariance(np.array([[v]]))
    assert np.isfinite(D).all(), f"D = {D} for a finite cov of {v}"
    assert float(D[0]) == v


@pytest.mark.parametrize("v", [1e308, float(np.finfo(float).max)])
def test_the_front_door_prices_a_huge_finite_variance(v):
    """It used to raise `D has a non-finite entry` for a covariance it
    had itself just accepted as finite."""
    p = np.asarray(race_probabilities([2.0], cov=[[v]]))
    assert p.shape == (1,)
    assert p[0] == 1.0


@pytest.mark.parametrize("v", [1e200, 1e300])
def test_a_large_two_runner_cov_survives_too(v):
    """Not a one-runner special case: the symmetrisation is on the
    shared path. At 1e308 a two-runner fit still overflows further in
    (`_center2` sums a column), which is a separate and much larger
    piece of work -- fitting a field that size needs the covariance
    rescaled before the fit, not one more guarded addition. This pins
    what the fix does reach."""
    C = np.array([[v, 0.0], [0.0, v]])
    V, D, F, W = fit_covariance(C)
    assert np.isfinite(D).all()


def test_an_overflowing_trace_does_not_disable_the_psd_check():
    """The tolerance was `-1e-8 * max(trace(C) / n, 1e-300)`, and
    trace overflows before the division. That makes the tolerance
    -inf, and `lam_min < -inf` is False for every lam_min, so the
    check ACCEPTED a matrix whose smallest eigenvalue is -5e307.
    Dividing before summing keeps the tolerance finite."""
    C = np.array([[1e308, 1.5e308], [1.5e308, 1e308]])
    assert np.linalg.eigvalsh(C).min() < 0          # it is not PSD
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_covariance(C)


def test_halving_first_is_the_same_number():
    """The fix must not move any answer that already worked."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        A = rng.standard_normal((5, 5)) * float(10.0 ** float(
            rng.integers(-200, 200)))
        assert np.array_equal(0.5 * (A + A.T), 0.5 * A + 0.5 * A.T)
