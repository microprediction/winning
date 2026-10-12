"""A sampling method's budget is a whole number of at least one draw (#668).

Zero draws came back as an all-NaN probability vector from five methods,
a fraction was truncated silently, and True ran one draw. The lattice
method's budget is a lattice size instead: None is the default, and 2.5
used to truncate to a two-point lattice.
"""
import numpy as np
import pytest

import winning.methods  # noqa: F401  (registers the methods)
from winning.methods.registry import METHODS, as_draws

MU = np.array([0.0, 1.0, 0.3])
V = np.empty((3, 0))
D = np.ones(3)

SAMPLERS = ["direct_mc", "sobol_direct", "factor_rqmc", "ghk", "qmc_ghk",
            "tilting", "stern", "genz_bretz", "smc_orthant"]


def test_every_method_with_a_draw_budget_is_listed():
    # a sampler added to the registry has to come through the validator
    # too: any registered method whose budget defaults to an integer is a
    # sampler, and must be in the list the tests below sweep
    import inspect
    found = sorted(
        name for name, fn in METHODS.items()
        if isinstance(inspect.signature(fn).parameters.get(
            "budget", inspect.Parameter("x", 1)).default, int))
    assert found == sorted(SAMPLERS), found


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("bad", [0, -3, 2.5, True, float("nan"),
                                 float("inf"), "8", None])
def test_a_sampler_refuses_a_budget_that_is_not_a_positive_whole_number(name, bad):
    with pytest.raises(ValueError, match="budget"):
        METHODS[name](MU, V, D, budget=bad)


@pytest.mark.parametrize("name", SAMPLERS)
def test_one_draw_is_a_legal_budget(name):
    p, _ = METHODS[name](MU, V, D, budget=1)
    assert p.shape == (3,) and not np.isnan(p).any()


@pytest.mark.parametrize("name", SAMPLERS)
def test_a_whole_float_or_numpy_integer_is_the_same_budget(name):
    a, _ = METHODS[name](MU, V, D, budget=64)
    b, _ = METHODS[name](MU, V, D, budget=64.0)
    c, _ = METHODS[name](MU, V, D, budget=np.int64(64))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_as_draws_returns_a_python_int():
    assert as_draws(np.int32(5)) == 5 and type(as_draws(5.0)) is int


@pytest.mark.parametrize("bad", [0, 1, 2.5, True, float("nan")])
def test_lattice_refuses_a_size_that_is_not_a_lattice(bad):
    with pytest.raises(ValueError, match="budget"):
        METHODS["lattice"](MU, V, D, budget=bad)


def test_lattice_default_and_explicit_size_agree():
    a, _ = METHODS["lattice"](MU, V, D)
    b, _ = METHODS["lattice"](MU, V, D, budget=None)
    c, _ = METHODS["lattice"](MU, V, D, budget=501)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
