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


@pytest.mark.parametrize("c", [0.1, 10.0])
def test_pair_closed_form_reads_relative_weights(c):
    """#170: W and c W are the same factor law; the forward pair closed form
    treated the absolute scale as mass (0.72 -> 0.61 at c = 10) while the
    inverse normalised, so the round trip 'converged' 5.6e-2 off."""
    mu = np.array([0.0, 1.0])
    V = np.array([[0.0], [1.0]])
    D = np.array([1.0, 1.0])
    F = np.array([[-1.0], [1.0]])
    W = np.array([0.5, 0.5])
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    pc = race_probabilities(mu, V=V, D=D, F=F, W=c * W)
    assert np.abs(pc - p).max() < 1e-14
    m, info = abilities_from_race(p, V=V, D=D, F=F, W=c * W, return_info=True)
    assert info["converged"]
    assert np.abs(race_probabilities(m, V=V, D=D, F=F, W=c * W) - p).max() < 1e-12
    # the general lattice at n = 3 has always read relative weights
    mu3, V3, D3 = np.array([0.0, 0.5, 1.0]), np.array([[0.0], [0.5], [1.0]]), np.ones(3)
    p3 = race_probabilities(mu3, V=V3, D=D3, F=F, W=W)
    assert np.abs(race_probabilities(mu3, V=V3, D=D3, F=F, W=c * W) - p3).max() < 1e-14


@pytest.mark.parametrize("n", [16, 30, 40])
def test_dense_correlation_inverse_converges(n):
    """#178: a dense short-length-scale correlation puts two runners 1e-5
    apart, whose contrast sd is 0.014 -- nearly unidentified, so the
    own-slope preconditioner (which sees each runner's own marginal, not
    the contrast) understates the step by the same factor every sweep.
    The iteration is monotone, riding one mode; extrapolation sums it.

    Before: the damping's undo-and-halve ratcheted to its floor on an
    early transient and never recovered, and the inverse reported
    non-convergence after 120 sweeps at 3.2e-1 of log residual -- while
    the same sweeps with no damping at all converged in 93."""
    from winning.factor import race_probabilities as _rp
    rng = np.random.default_rng(n)
    x = np.sort(rng.random(n))
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / 0.15)
    mu0 = np.linspace(-0.5, 0.5, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = _rp(mu0, cov=C, points=257)
        mu, info = abilities_from_race(p, cov=C, points=257, return_info=True)
        q = _rp(mu, cov=C, points=257)
    assert info["converged"], info
    assert info["iterations"] <= 100
    assert np.abs(q - p).max() < 1e-8


def test_caution_is_temporary_but_the_richardson_value_is_not():
    """#178: the two must not share one number. A sweep that fails to
    contract is a transient and its penalty is restored on the next good
    sweep; the Richardson damping is a fact about the Jacobian that a
    good sweep does not repeal. Restoring THAT is what stops the
    heterogeneous cases converging, so this pins both directions at once:
    the dense field needs the recovery, the heterogeneous pair needs the
    Richardson value to stick."""
    from winning.factor import race_probabilities as _rp
    rng = np.random.default_rng(30)
    x = np.sort(rng.random(30))
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / 0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = _rp(np.linspace(-0.5, 0.5, 30), cov=C, points=257)
        needs_recovery = abilities_from_race(p, cov=C, points=257,
                                             return_info=True)[1]
    needs_sticky = abilities_from_race(np.array([0.6, 0.2, 0.2]),
                                       D=np.array([0.03, 0.03, 1.0]),
                                       points=500, return_info=True)[1]
    assert needs_recovery["converged"] and needs_sticky["converged"]
    assert needs_sticky["iterations"] <= 25
