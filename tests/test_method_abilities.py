"""All registered methods reject invalid locations before shortcuts (#914)."""
import numpy as np
import pytest

import winning.methods  # noqa: F401
from winning.methods.registry import METHODS


BAD_MU = [[], [np.nan], [np.inf], [-np.inf],
          [np.nan, 0.0], [0.0, np.inf], [0.0, -np.inf],
          [0.0, np.nan, 0.3], 0.0, [[0.0], [0.3]]]


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("mu", BAD_MU)
def test_invalid_abilities_rejected_before_method_shortcuts(name, mu):
    n = np.asarray(mu).size
    with pytest.raises(ValueError, match="mu"):
        METHODS[name](mu, np.empty((n, 0)), np.ones(n), budget=8, seed=7)


@pytest.mark.parametrize("mu", BAD_MU)
def test_explicit_covariance_does_not_bypass_ability_validation(mu):
    with pytest.raises(ValueError, match="mu"):
        METHODS["qmc_ghk"](mu, None, None, cov=np.eye(2), budget=8, seed=7)


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("mu", [[0.3], [0.0, 0.4], [0.0, 0.4, -0.2]])
def test_valid_abilities_preserve_method_output_and_inputs(name, mu):
    mu = np.array(mu)
    n = len(mu)
    V, D = np.full((n, 1), 0.2), np.ones(n)
    original = [a.copy() for a in (mu, V, D)]
    wrapped, _ = METHODS[name](mu, V, D, budget=32, seed=7)
    direct, _ = METHODS[name].__wrapped__(mu, V, D, budget=32, seed=7)
    np.testing.assert_array_equal(wrapped, direct)
    for a, before in zip((mu, V, D), original):
        np.testing.assert_array_equal(a, before)


def test_valid_explicit_covariance_keeps_shape_bypass():
    mu, cov = np.array([0.0, 0.4]), np.array([[1.0, 0.2], [0.2, 2.0]])
    wrapped, _ = METHODS["qmc_ghk"](mu, None, None, cov=cov, budget=32)
    direct, _ = METHODS["qmc_ghk"].__wrapped__(mu, None, None, cov=cov, budget=32)
    np.testing.assert_array_equal(wrapped, direct)


def test_a_registered_method_receives_the_callers_mu():
    # validation runs on a checked copy: a custom method that is handed a
    # tuple today still gets the tuple (#914 review)
    from winning.methods.registry import register
    seen = []

    @register("_test_passthrough_914")
    def passthrough(mu, V, D, **kw):
        seen.append(mu)
        return np.full(len(mu), 1.0 / len(mu)), {}
    try:
        mu = (0.0, 0.4)
        METHODS["_test_passthrough_914"](mu, np.empty((2, 0)), np.ones(2))
        assert seen[0] is mu
    finally:
        METHODS.pop("_test_passthrough_914", None)


@pytest.mark.parametrize("mu", [np.ma.array([0.1], mask=True),
                                np.ma.array([0.1, 0.2], mask=[False, True]),
                                np.array([0.1 + 0.0j]), np.array([0.0, 0.3 + 2j])])
def test_masked_and_complex_abilities_are_refused_before_conversion(mu):
    # np.asarray(..., float) kept a masked payload and dropped an imaginary
    # part, so direct_mc priced [0.1] as the certain race [1.]
    n = np.asarray(mu).size
    with pytest.raises(ValueError, match="mu"):
        METHODS["direct_mc"](mu, np.empty((n, 0)), np.ones(n), budget=8, seed=7)
    from winning.factor.races import race_probabilities
    with pytest.raises(ValueError, match="mu"):
        race_probabilities(mu)


def test_an_unmasked_masked_array_is_its_data():
    from winning.factor.races import race_probabilities
    mu = np.ma.array([0.0, 0.4, -0.2], mask=False)
    np.testing.assert_array_equal(race_probabilities(mu), race_probabilities(mu.data))
