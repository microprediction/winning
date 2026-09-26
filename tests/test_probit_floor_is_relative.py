"""The variance floor scales with the covariance it floors.

A race is unchanged by measuring performances in different units: scale
every utility by c and every variance by c**2 and the shares are the
same numbers. The `D=` path had that exactly. The `Sigma=` path did not,
because `factor_model_projected` floored D at an ABSOLUTE 1e-8, so
`Sigma = 1e-12 * I` came back as `D = 1e-8` -- the variance inflated ten
thousandfold -- and the shares collapsed toward uniform (#101).

`fit_covariance` had already settled this question in the same file:
"the floor is RELATIVE to each runner's own variance, not an absolute
multiple of the mean ... An absolute floor destroys near-singular
contrasts". The fix was made in one of the two places this file floors a
variance and not the other.
"""
import numpy as np
import pytest

from winning.probit import fit_factor_model, shares

U = np.array([0.8, 0.1, -0.4])
U = U - U.mean()


@pytest.mark.parametrize("c", [1.0, 1e-2, 1e-4, 1e-6, 1e-9])
def test_shares_are_invariant_to_the_unit_of_performance(c):
    ref = shares(U, D=np.ones(3))
    got = shares(c * U, Sigma=c * c * np.eye(3), k=1)
    assert np.allclose(got, ref, atol=1e-12), f"scale {c:.0e} moved the race"


@pytest.mark.parametrize("c", [1.0, 1e-4, 1e-9])
def test_the_two_descriptions_of_one_race_agree(c):
    """`D=` and `Sigma=` describe the same independent race here, so any
    gap between them is the fitter's, not the model's."""
    a = shares(c * U, D=c * c * np.ones(3))
    b = shares(c * U, Sigma=c * c * np.eye(3), k=1)
    assert np.allclose(a, b, atol=1e-12)


def test_the_fitted_variance_is_the_one_supplied():
    """The symptom underneath: D came back 1e4 times the input."""
    for c in (1.0, 1e-4, 1e-6):
        V, D = fit_factor_model(c * c * np.eye(3), 1)
        assert np.allclose(D, c * c, rtol=1e-9), (
            f"Sigma = {c * c:.1e} * I fitted D = {D[0]:.1e}")
        assert np.allclose(V, 0.0, atol=1e-12), "a diagonal has no factor"


def test_a_near_singular_contrast_still_survives():
    """The case the sibling floor's comment was written for: an absolute
    floor 'destroys near-singular contrasts'. diag(1e-8, 1e-8, 1) must
    keep Var(X1 - X2) at 2e-8, not raise it."""
    V, D = fit_factor_model(np.diag([1e-8, 1e-8, 1.0]), 1)
    assert D[0] + D[1] == pytest.approx(2e-8, rel=1e-6)


def test_the_floor_still_keeps_every_variance_positive():
    """Relative must not mean absent: a zero variance still gets floored
    to something positive, or a downstream sqrt/divide breaks."""
    V, D = fit_factor_model(np.diag([0.0, 1.0, 1.0]), 1)
    assert (D > 0).all(), D


def test_unit_scale_is_untouched():
    """READ OFF the pre-fix code: the multiplier is unchanged, so at
    unit scale this must be the same floor it always was. (Invented
    twice in this session before being measured -- a pin whose value
    was guessed pins nothing.)"""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    C = A @ A.T / 5 + np.eye(5)
    V, D = fit_factor_model(C, 2)
    assert float(D.sum()) == pytest.approx(4.8501821493, abs=1e-8)
