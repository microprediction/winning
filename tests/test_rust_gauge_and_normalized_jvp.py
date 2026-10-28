"""The compiled kernels take the same gauge-fixed loadings and the same
normalisation as the numpy spec (#114, #124).

The kernel does not center V; a common loading column (the same shock to
every runner) cannot move an argmin, and the numpy path subtracts it
before building the lattice window. The compiled dispatch used to pass V
uncentered, so a shift of 100 moved the answer by 3e-2 (forward) and
1e-1 (JVP) with rust on and 3e-16 with it off. The compiled JVP also
returned the raw derivative of the unnormalised rectangle sum where the
public function promises the normalised one: at 25 lattice points the two
differ by 3.5e-4 (the existing parity test ran at 3001 points, where the
total mass is 1 to 1e-13 and the quotient rule is invisible)."""
import numpy as np
import pytest

import winning
from winning.factor.core import (hermite_nodes, jacobian_vector_product,
                                 win_probabilities_factor)
import winning.methods.native as native

pytestmark = pytest.mark.skipif(not winning.rust_active(),
                                reason="needs the compiled kernels")


@pytest.fixture
def rust_on_after():
    yield
    winning.use_rust(True)


def _field():
    mu = np.array([-1.0, -0.3, 0.2, 0.7, 1.4])
    V = np.array([[-0.5], [-0.2], [0.0], [0.3], [0.4]])
    D = np.ones(5)
    F, W = hermite_nodes(1, Q=15)
    h = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
    return mu, V, D, F, W, h


@pytest.mark.parametrize("shift", [100.0, -37.5])
def test_compiled_forward_is_invariant_to_a_common_loading(shift, rust_on_after):
    mu, V, D, F, W, _ = _field()
    winning.use_rust(True)
    p0 = win_probabilities_factor(mu, V, D, F, W, points=501)
    p1 = win_probabilities_factor(mu, V + shift, D, F, W, points=501)
    assert np.abs(p0 - p1).max() < 1e-12


@pytest.mark.parametrize("shift", [100.0, -37.5])
def test_compiled_jvp_is_invariant_to_a_common_loading(shift, rust_on_after):
    mu, V, D, F, W, h = _field()
    winning.use_rust(True)
    j0 = jacobian_vector_product(mu, V, D, F, W, h, points=501)
    j1 = jacobian_vector_product(mu, V + shift, D, F, W, h, points=501)
    assert np.abs(j0 - j1).max() < 1e-12


def test_methods_lattice_is_invariant_to_a_common_loading(rust_on_after):
    mu, V, D, _, _, _ = _field()
    winning.use_rust(True)
    p0, i0 = native.lattice(mu, V, D, budget=501)
    p1, i1 = native.lattice(mu, V + 100.0, D, budget=501)
    assert i0["backend"] == "fastrace"
    assert np.abs(p0 - p1).max() < 1e-12


@pytest.mark.parametrize("points", [25, 101, 3001])
@pytest.mark.parametrize("form", ["ibp", "grid"])
def test_compiled_jvp_is_the_normalised_derivative(points, form, rust_on_after):
    """#124's own field: centered loadings, five nodes, a coarse lattice
    where the unnormalised total is not 1 and the quotient rule matters."""
    rng = np.random.default_rng(0)
    K = 5
    mu = rng.normal(size=K)
    V = rng.normal(size=(K, 1)) * .5
    V -= V.mean(axis=0)
    D = .6 + rng.random(K)
    F, W = hermite_nodes(1, Q=5)
    h = rng.normal(size=K)
    winning.use_rust(True)
    j_rust = jacobian_vector_product(mu, V, D, F, W, h, points=points, form=form)
    winning.use_rust(False)
    j_np = jacobian_vector_product(mu, V, D, F, W, h, points=points, form=form)
    j_raw = jacobian_vector_product(mu, V, D, F, W, h, points=points, form=form,
                                    normalized=False)
    assert np.abs(j_rust - j_np).max() < 1e-12 * max(1.0, np.abs(j_np).max())
    assert abs(j_rust.sum()) < 1e-12               # tangent to the simplex
    if points == 25:
        # the raw derivative is a different vector here; the test is live
        assert np.abs(j_raw - j_np).max() > 1e-5
