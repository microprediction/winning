"""A supplied Sigma goes through the same door as cov=.

`winning.probit` fitted whatever it was handed straight through
`fit_factor_model`. An indefinite matrix (eigenvalue -1) returned
shares of [1, 0] and an asymmetric one a plausible [0.649, 0.351],
neither with a warning (#102). A factor fit of a matrix that is not a
covariance has nothing to be checked against, so the check belongs
before the fit rather than after it.
"""
import numpy as np
import pytest

from winning.factor.core import _validate_covariance
from winning.probit import removal_shares, shares, utilities_from_shares

U = [0.2, -0.2]
BAD = {
    "indefinite": [[1.0, 2.0], [2.0, 1.0]],
    "asymmetric": [[1.0, 0.8], [0.1, 1.0]],
    "non-finite": [[1.0, np.nan], [np.nan, 1.0]],
    "non-square": [[1.0, 0.5]],
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_shares_refuses_a_matrix_that_is_not_a_covariance(name):
    with pytest.raises(ValueError, match="Sigma"):
        shares(U, Sigma=BAD[name], k=1)


@pytest.mark.parametrize("name", sorted(BAD))
def test_every_probit_entry_point_refuses_it(name):
    """`_prepare` is the one place Sigma enters, so all three verbs
    inherit the check; a test per verb is what proves that rather than
    assuming it."""
    with pytest.raises(ValueError, match="Sigma"):
        utilities_from_shares([0.6, 0.4], Sigma=BAD[name], k=1)
    with pytest.raises(ValueError, match="Sigma"):
        removal_shares(U, Sigma=BAD[name], k=1)


def test_the_message_names_the_argument_the_caller_passed():
    """`cov=` would be wrong here -- the caller wrote Sigma=."""
    with pytest.raises(ValueError) as e:
        shares(U, Sigma=BAD["indefinite"], k=1)
    assert "Sigma" in str(e.value) and "cov=" not in str(e.value)


def test_a_valid_sigma_is_unaffected():
    p = shares(U, Sigma=[[1.0, 0.0], [0.0, 1.0]], k=1)
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[1]
    # and a correlated one still fits
    q = shares(U, Sigma=[[1.0, 0.5], [0.5, 1.0]], k=1)
    assert np.isclose(q.sum(), 1.0)
    assert not np.allclose(p, q), "the correlation must move the shares"


@pytest.mark.parametrize("name", sorted(BAD))
def test_sigma_and_cov_enforce_the_same_contract(name):
    """The defect was two doors onto one question with only one of them
    asking it, so the claim is that both now refuse the SAME matrices
    for the SAME reason -- differing only in the argument they name.

    Asserted behaviourally. An earlier version of this test read
    `inspect.getsource(fit_covariance)` for the call, which is the
    marker-inspection anti-pattern: inlining the identical checks would
    have failed it, and calling the validator while ignoring its result
    would have passed it.
    """
    from winning.factor.races import race_probabilities

    with pytest.raises(ValueError) as via_sigma:
        shares(U, Sigma=BAD[name], k=1)
    with pytest.raises(ValueError) as via_cov:
        race_probabilities(np.zeros(2), cov=BAD[name])
    a = str(via_sigma.value).replace("Sigma", "<arg>")
    b = str(via_cov.value).replace("cov=", "<arg>")
    assert a == b, f"same input, different complaint:\n  {a}\n  {b}"


def test_the_validator_returns_the_symmetrised_matrix():
    C = _validate_covariance([[2.0, 0.5], [0.5, 1.5]])
    assert np.allclose(C, C.T)
