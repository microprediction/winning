"""The two-runner closed form under a caller's NON-centred node law.

Runner i performs mu_i + V_i . F + e_i (lowest wins), so with F drawn
from the caller's nodes the pair is an exact mixture of normal
contrasts. The closed form added the node mean to the contrast instead
of subtracting it: a node law with mean 1.43 priced an exact
[0.128, 0.872] as [0.936, 0.064] (found while porting #86 to the
browser). The default rules and covariance fits supply centred nodes,
so ordinary calls never reached the wrong sign.

The closed form matches the node law's mean and covariance, so it is
exact when the nodes are a quadrature of a NORMAL law (every built-in
rule); the reference cases here are therefore Gauss-Hermite rules for
shifted, scaled normals. (An arbitrary discrete node law is matched only
to its first two moments; that approximation is a separate question.)
"""
import numpy as np
import pytest
from scipy.stats import norm

from winning.factor import abilities_from_race, race_probabilities


def _exact_pair(mu, V, D, F, W):
    """P(runner 0 lowest), node by node: an independent reference."""
    w = np.asarray(W, float) / np.sum(W)
    s = np.sqrt(D[0] + D[1])
    gaps = (mu[1] + F @ V[1]) - (mu[0] + F @ V[0])
    return float(w @ norm.cdf(gaps / s))


def _gh_law(m, L, order=24):
    """Tensor Gauss-Hermite nodes and weights for N(m, L L')."""
    x, w = np.polynomial.hermite_e.hermegauss(order)
    r = len(m)
    grids = np.meshgrid(*([x] * r), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    wg = np.meshgrid(*([w] * r), indexing="ij")
    wts = np.prod(np.stack([g.ravel() for g in wg], axis=1), axis=1)
    return m + Z @ L.T, wts / wts.sum()


def _cases():
    rng = np.random.default_rng(86)
    for _ in range(12):
        r = int(rng.integers(1, 3))
        mu = rng.normal(size=2)
        V = rng.normal(size=(2, r))
        D = rng.uniform(0.3, 2.0, size=2)
        m = rng.normal(size=r) * 1.5                      # off-centre
        L = np.diag(rng.uniform(0.4, 1.2, size=r))
        F, W = _gh_law(m, L, order=80 if r == 1 else 48)
        yield mu, V, D, F, W


@pytest.mark.parametrize("case", list(_cases()))
def test_pair_closed_form_matches_the_exact_mixture(case):
    mu, V, D, F, W = case
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    want = _exact_pair(mu, V, D, F, W)
    np.testing.assert_allclose(p, [want, 1.0 - want], rtol=1e-9, atol=1e-12)


def test_the_reported_case():
    # a discrete three-node law with mean 1.43: exact 0.127732; the closed
    # form matches two moments (0.126806) and gave 0.936016 before
    mu = np.array([0.0, 0.3])
    V = np.array([[1.0], [-0.5]])
    D = np.array([1.0, 1.0])
    F = np.array([[0.8], [1.4], [2.0]])
    W = np.array([0.3, 0.4, 0.3])
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    assert abs(p[0] - _exact_pair(mu, V, D, F, W)) < 2e-3


@pytest.mark.parametrize("case", list(_cases())[:6])
def test_pair_agrees_with_the_general_lattice(case):
    # a third runner far behind (min-wins: far ABOVE) forces the general
    # lattice on the same node law; it must agree to lattice accuracy
    mu, V, D, F, W = case
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    mu3 = np.r_[mu, np.max(np.abs(mu)) + 60.0]
    V3 = np.vstack([V, np.zeros(V.shape[1])])
    lat = race_probabilities(mu3, V=V3, D=np.r_[D, 1.0], F=F, W=W)
    assert lat[2] < 1e-30
    np.testing.assert_allclose(p, lat[:2], atol=2e-3)


@pytest.mark.parametrize("case", list(_cases())[:6])
def test_pair_forward_and_inverse_round_trip(case):
    mu, V, D, F, W = case
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    if p.min() < 1e-6:
        pytest.skip("near-certain pair: the gap is not well determined")
    got = abilities_from_race(p, V=V, D=D, F=F, W=W)
    np.testing.assert_allclose(got[1] - got[0], mu[1] - mu[0],
                               rtol=1e-8, atol=1e-8)


def test_centred_nodes_are_unchanged():
    # the shift is zero for a centred law, so those calls keep their bits
    rng = np.random.default_rng(7)
    mu, V, D = rng.normal(size=2), rng.normal(size=(2, 1)), np.ones(2)
    F, W = _gh_law(np.zeros(1), np.eye(1))
    np.testing.assert_allclose(race_probabilities(mu, V=V, D=D, F=F, W=W)[0],
                               _exact_pair(mu, V, D, F, W), rtol=1e-9)
