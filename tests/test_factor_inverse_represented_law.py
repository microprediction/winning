"""The direct factor inverse (winning.factor.core) inverts the map the
forward evaluates: the represented factor law, not raw loadings (#443),
and the derivative of the NORMALISED share (#371)."""
import numpy as np
import pytest

from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes, win_probabilities_factor)


def _p(r):
    return r[0] if isinstance(r, tuple) else r


@pytest.mark.parametrize("a", [1e-4, 1.0, 100.0, 1e4])
def test_reciprocal_loading_and_node_scale_is_the_same_inverse(a):
    # V -> aV, F -> F/a leaves mu + F V' unchanged; at a = 100 the inverse
    # used to miss its own target by 89 points (#443)
    mu0 = np.array([-1.2, -0.1, 0.4, 0.9])
    V0 = np.array([[-1.0], [-0.2], [0.4], [0.8]])
    D = np.array([0.3, 0.8, 0.5, 1.1])
    F0, W = np.array([[-1.0], [1.0]]), np.array([0.5, 0.5])
    t = _p(win_probabilities_factor(mu0, V0, D, F0, W, points=1001))
    V, F = a * V0, F0 / a
    mu, info = abilities_from_probabilities_factor(
        t, V, D, F, W, points=501, n_iter=50, tol=1e-8, return_info=True)
    assert info["converged"], info
    back = _p(win_probabilities_factor(mu, V, D, F, W, points=4001))
    assert np.abs(back - t).max() < 1e-7


def test_coarse_lattice_self_round_trip():
    # 21 points: the raw own-slope over the normalised share parted from
    # finite differences by 85% and the solve missed by 22 points (#371)
    mu = np.array([.771, .170, -.474, -.467])
    D = np.array([1.426, .379, .415, .363])
    V = np.array([[-1.946], [.471], [-2.087], [3.562]])
    F = np.array([[-2.02018287045609], [-.958572464613819], [0.0],
                  [.958572464613819], [2.02018287045609]])
    W = np.array([.0112574113277207, .222075922005613, .533333333333333,
                  .222075922005613, .0112574113277207])
    t = _p(win_probabilities_factor(mu, V, D, F, W, points=21))
    m, info = abilities_from_probabilities_factor(
        t, V, D, F, W, points=21, n_iter=50, tol=1e-8, return_info=True)
    assert info["converged"], info
    assert np.abs(_p(win_probabilities_factor(m, V, D, F, W, points=21)) - t).max() < 1e-7


def test_the_inverse_does_not_depend_on_the_loading_gauge():
    # a dominant favourite: the old step cap read raw row norms, so it
    # converged for one gauge of V and stalled at 3e-2 for the centred one
    rng = np.random.default_rng(3)
    F, W = hermite_nodes(2)
    mu0 = rng.normal(0, 1, 3)
    mu0 -= mu0.mean()
    V = rng.normal(0.0, 0.6, (3, 2))
    D = rng.uniform(0.5, 2.0, 3)
    p = _p(win_probabilities_factor(mu0, V, D, F, W))
    for VV in (V, V - V.mean(axis=0)):
        mu, info = abilities_from_probabilities_factor(
            p / p.sum(), VV, D, F, W, return_info=True)
        assert info["converged"], info
        assert np.abs(mu - mu0).max() < 1e-5


def test_pair_closed_form_reads_the_rule_mean():
    # a translated two-point rule: Gaussian-free, but its mean shifts the
    # pair contrast by (v1 - v0) E[F]
    V = np.array([[0.0], [1.0]])
    D = np.array([1.0, 1.0])
    F, W = hermite_nodes(1, Q=31)
    F = F + 1.0
    t = _p(win_probabilities_factor(np.array([-0.5, 0.5]), V, D, F, W))
    mu = abilities_from_probabilities_factor(t, V, D, F, W)
    assert np.abs(mu - np.array([-0.5, 0.5])).max() < 1e-6
