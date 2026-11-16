"""top_k_jacobians' dq/dsigma includes the moving rule weights (#783).

The automatic rank-one factor rule has fixed nodes but weights that are
smooth functions of the sharpness sqrt(2) max_i |V_i| / sigma_i: the order
interpolation and pair of #850 below 9, the Gauss-Hermite/midpoint blend
of #783 on (9, 10]. The Jacobian used to differentiate only the node
mixture, so inside those intervals Jsigma missed
(dsharp/dsigma_j) sum_q (dW_q/dsharp) q_ind(mu + V f_q) and sat far from
the forward's own finite difference (sharpness 8.9: -0.0024 against
+0.187). Fixture and sharpness points from the issue thread.
"""
import numpy as np
import pytest

from winning.factor import races
from winning.factor.topk import top_k_jacobians, top_k_probabilities

MU = np.array([-0.7, -0.1, 0.2, 0.8])
K = 2


def _V(sharp):
    a = sharp / np.sqrt(2)
    return np.array([[-a], [a / 3], [a / 3], [a / 3]])   # runner 0 sets it


def _fd_sigma(mu, k, V, D, j, h=1e-6, qa=None):
    def p(e):
        Dj = D.copy()
        Dj[j] = (np.sqrt(D[j]) + e) ** 2
        return top_k_probabilities(mu, k, V=V, D=Dj, qa=qa)
    return (p(h) - p(-h)) / (2 * h)


@pytest.mark.parametrize("sharp", [5.5, 8.95, 9.5])
def test_jsigma_matches_finite_differences_inside_moving_weights(sharp):
    """5.5: the sliding Gauss-Hermite pair; 8.95: the hand-back to GH(72);
    9.5: the Gauss-Hermite/midpoint blend."""
    V, D = _V(sharp), np.ones(4)
    _, Js = top_k_jacobians(MU, K, D=D, V=V)
    for j in range(4):
        fd = _fd_sigma(MU, K, V, D, j)
        assert np.max(np.abs(Js[:, j] - fd)) < 1e-7, (sharp, j)


@pytest.mark.parametrize("sharp", [5.5, 8.95, 9.5])
def test_term_lands_on_the_runner_that_sets_the_sharpness(sharp):
    """A random field whose sharpest runner is not runner 0."""
    rng = np.random.default_rng(783)
    mu = rng.normal(size=5)
    v = rng.normal(size=5)
    D = np.exp(0.5 * rng.normal(size=5))
    vc = v - v.mean()
    ratio = np.abs(vc) / np.sqrt(D)
    V = (v * sharp / (np.sqrt(2) * ratio.max()))[:, None]
    j = int(np.argmax(ratio))
    assert j != 0
    _, Js = top_k_jacobians(mu, K, D=D, V=V)
    fd = _fd_sigma(mu, K, V, D, j)
    assert np.max(np.abs(Js[:, j] - fd)) < 1e-7


def test_no_term_where_the_rule_is_constant_or_fixed():
    """Outside the moving intervals, and for an explicit qa=, Jsigma is the
    node mixture alone (the helper finds nothing to add)."""
    D = np.ones(4)
    for sharp in (1.0, 4.0, 7.5, 12.0):
        V = races._gauge_center(_V(sharp))
        assert races._rank1_rule_sigma_slope(V, D) is None, sharp
    V = _V(9.5)
    _, Js = top_k_jacobians(MU, K, D=D, V=V, qa=15)
    fd = _fd_sigma(MU, K, V, D, 0, qa=15)
    assert np.max(np.abs(Js[:, 0] - fd)) < 1e-7


@pytest.mark.parametrize("sharp", [2.3, 3.05, 5.5, 8.95, 9.5])
def test_rule_slope_is_the_derivative_of_the_rule(sharp):
    """dW/dsharp from the rule builders against a difference of W."""
    rule = (races._rank1_blend_rule if sharp > races._R1_BLEND_LO
            else races._rank1_hermite_rule)
    F, W, dW = rule(sharp, slope=True)
    h = 1e-6
    Fp, Wp = rule(sharp + h)
    Fm, Wm = rule(sharp - h)
    assert np.array_equal(Fp, F) and np.array_equal(Fm, F)
    assert np.max(np.abs((Wp - Wm) / (2 * h) - dW)) < 1e-6
    F0, W0 = rule(sharp)
    assert np.array_equal(F0, F) and np.array_equal(W0, W)


def test_tie_for_the_max_goes_to_the_first_runner():
    """Two runners share the max, a kink of the sharpness: the term goes
    to the first of them, giving its one-sided derivative as its sd
    falls; the other's column is the node mixture alone, its one-sided
    derivative as its sd rises (the first runner then holds the max)."""
    sharp = 9.5
    a = sharp / np.sqrt(2)
    V = np.array([[-a], [a], [0.0], [0.0]])
    D = np.ones(4)
    found = races._rank1_rule_sigma_slope(races._gauge_center(V), D)
    assert found is not None and found[0] == 0
    _, Js = top_k_jacobians(MU, K, D=D, V=V)
    p0 = top_k_probabilities(MU, K, V=V, D=D)
    h = 1e-6

    def p(j, sd):
        Dj = D.copy()
        Dj[j] = sd * sd
        return top_k_probabilities(MU, K, V=V, D=Dj)
    assert np.max(np.abs(Js[:, 0] - (p0 - p(0, 1 - h)) / h)) < 1e-5
    assert np.max(np.abs(Js[:, 1] - (p(1, 1 + h) - p0) / h)) < 1e-5
