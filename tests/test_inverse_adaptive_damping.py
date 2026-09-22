"""The inverse damps by the contraction it observes, reads its scale off
the nodes it is given, and abilities_from_topk(k=1) shares both (#149,
#151).

Two cases the fixed top-two-share gate could not see: heterogeneous
variances make the same negative Jacobi eigenvalue with a share of 0.8
exactly (p = [.6, .2, .2], D = [.03, .03, 1]: 60 sweeps and 1.7e-4
short undamped, 17 at 0.7) and with a share of 0.75 (p = [.45, .3,
.25], D = [.1, 30, 1]). And (V, F) -> (V / c, c F) is the identical
forward map, but the warm-start scale read off V alone was 33x off at
c = 0.03 and the inverse returned [0, 4e-10, 1] for [.6, .25, .15]."""
import warnings

import numpy as np
import pytest

from winning.factor import abilities_from_race, race_probabilities
from winning.factor.core import hermite_nodes
from winning.factor.topk import abilities_from_topk, top_k_probabilities


@pytest.mark.parametrize("p,D", [
    ([0.6, 0.2, 0.2], [0.03, 0.03, 1.0]),
    ([0.45, 0.30, 0.25], [0.1, 30.0, 1.0]),
])
def test_heterogeneous_variances_converge(p, D):
    p, D = np.array(p), np.array(D)
    mu, info = abilities_from_race(p, D=D, points=500, return_info=True)
    q = race_probabilities(mu, D=D, points=500)
    assert info["converged"], info
    assert info["iterations"] <= 30
    assert np.abs(q - p).max() < 1e-7


@pytest.mark.parametrize("c", [0.03, 1.0, 30.0])
def test_rescaling_the_nodes_and_loadings_together_changes_nothing(c):
    F, W = hermite_nodes(1, Q=15)
    V = np.array([[-.5], [0.0], [.5]])
    D = np.ones(3)
    p = np.array([.6, .25, .15])
    ref, info_ref = abilities_from_race(p, V=V, D=D, F=F, W=W, return_info=True)
    mu, info = abilities_from_race(p, V=V / c, D=D, F=c * F, W=W, return_info=True)
    assert info_ref["converged"] and info["converged"], (info_ref, info)
    assert info["iterations"] == info_ref["iterations"]
    assert np.abs(mu - ref).max() < 1e-9
    q = race_probabilities(mu, V=V / c, D=D, F=c * F, W=W)
    assert np.abs(q - p).max() < 1e-8


def test_pair_closed_form_reads_the_represented_node_covariance():
    F, W = hermite_nodes(1, Q=15)
    V = np.array([[-.7], [.7]])
    D = np.array([.2, .3])
    p = np.array([.7, .3])
    ref = abilities_from_race(p, V=V, D=D, F=F, W=W)
    for c in (0.03, 30.0):
        mu = abilities_from_race(p, V=V / c, D=D, F=c * F, W=W)
        assert np.abs(mu - ref).max() < 1e-12
        assert np.abs(race_probabilities(mu, V=V / c, D=D, F=c * F, W=W) - p).max() < 1e-9


@pytest.mark.parametrize("n,eps", [(3, 6e-4), (10, 1e-4), (30, 1e-4)])
def test_topk_k1_effective_pair_converges(n, eps):
    """#151: abilities_from_topk(q, 1) is documented as abilities_from_race
    and kept the `n > 2` gate after #150 fixed the win race."""
    p = np.r_[np.linspace(.6, .4, 2), np.full(n - 2, eps)]
    p /= p.sum()
    D = np.full(n, 3.2 ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu, info = abilities_from_topk(p, 1, D=D, points=500, return_info=True)
    q = np.asarray(top_k_probabilities(mu, 1, D=D, points=500))
    assert info["converged"], info
    assert np.abs(q - p).max() < 1e-7


def test_topk_k2_is_unchanged_in_kind():
    """k >= 2 spreads the mass across slots; the k = 1 gate must not
    reach it, and the ordinary field still converges in a few sweeps."""
    rng = np.random.default_rng(1)
    mu0 = rng.normal(size=8)
    D = 0.5 + rng.random(8)
    q = np.asarray(top_k_probabilities(mu0, 2, D=D, points=257))
    mu, info = abilities_from_topk(q, 2, D=D, points=257, return_info=True)
    assert info["converged"] and info["iterations"] <= 20, info
    assert np.abs(mu - (mu0 - mu0.mean())).max() < 1e-6
