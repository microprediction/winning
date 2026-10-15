"""The two-runner closed form under a caller's NON-centred node law.

Runner i performs mu_i + V_i . F + e_i (lowest wins), so with F drawn
from the caller's nodes the pair is an exact mixture of normal
contrasts. The closed form added the node mean to the contrast instead
of subtracting it: a node law with mean 1.43 priced an exact
[0.128, 0.872] as [0.936, 0.064] (found while porting #86 to the
browser). The default rules and covariance fits supply centred nodes,
so ordinary calls never reached the wrong sign.

Under a caller's nodes the pair is now that exact mixture (#742): a
closed form matching only the node law's mean and covariance priced a
discrete law 1e-3 off, and a centred two-point law 0.5 for an exact
0.75. The default rules and covariance fits keep the closed form.
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
    # a discrete three-node law with mean 1.43: exact 0.127732; the
    # two-moment closed form gave 0.126806 (#742) and 0.936016 before
    mu = np.array([0.0, 0.3])
    V = np.array([[1.0], [-0.5]])
    D = np.array([1.0, 1.0])
    F = np.array([[0.8], [1.4], [2.0]])
    W = np.array([0.3, 0.4, 0.3])
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    assert abs(p[0] - _exact_pair(mu, V, D, F, W)) < 1e-14


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
    np.testing.assert_allclose(p, lat[:2], atol=1e-8)


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


def _discrete_cases():
    """Non-normal caller laws: the shapes two moments cannot see (#742)."""
    # the review's centred, asymmetric two-point law
    yield (np.zeros(2), np.array([[1.0], [0.0]]), np.array([0.01, 0.01]),
           np.array([[-1.0], [3.0]]), np.array([0.75, 0.25]))
    rng = np.random.default_rng(742)
    for q in (2, 3, 7, 40):
        r = int(rng.integers(1, 4))
        F = rng.exponential(size=(q, r)) - rng.uniform(0, 2, size=r)
        yield (rng.normal(size=2), rng.normal(size=(2, r)),
               rng.uniform(0.2, 1.5, size=2), F, rng.uniform(0.1, 3.0, size=q))


@pytest.mark.parametrize("case", list(_discrete_cases()))
def test_discrete_node_law_is_the_exact_mixture(case):
    mu, V, D, F, W = case
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    want = _exact_pair(mu, V, D, F, W)
    np.testing.assert_allclose(p, [want, 1.0 - want], rtol=1e-13, atol=1e-14)
    # relisting the runners relists the answer exactly
    q = race_probabilities(mu[::-1], V=V[::-1], D=D[::-1], F=F, W=W)
    assert np.array_equal(q, p[::-1])
    # a far-behind third runner forces the general lattice on the same law
    mu3 = np.r_[mu, np.max(np.abs(mu)) + 60.0]
    V3 = np.vstack([V, np.zeros(V.shape[1])])
    lat = race_probabilities(mu3, V=V3, D=np.r_[D, 1.0], F=F, W=W)
    np.testing.assert_allclose(p, lat[:2], atol=1e-8)


@pytest.mark.parametrize("case", list(_discrete_cases()))
def test_discrete_node_law_slopes(case):
    mu, V, D, F, W = case
    p, sl = race_probabilities(mu, V=V, D=D, F=F, W=W, return_slopes=True)
    h = 1e-6
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        fd = (race_probabilities(mu + e, V=V, D=D, F=F, W=W)[i]
              - race_probabilities(mu - e, V=V, D=D, F=F, W=W)[i]) / (2 * h)
        np.testing.assert_allclose(sl[i], fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("case", list(_discrete_cases())[1:])
def test_discrete_node_law_inverse_round_trips(case):
    # the inverse solves the mixture itself (a 1-D monotone root), so it
    # reprices its target to rounding, not to a two-moment approximation
    mu, V, D, F, W = case
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    got, info = abilities_from_race(p, V=V, D=D, F=F, W=W, return_info=True)
    assert info["converged"] and info["iterations"] == 0
    np.testing.assert_allclose(got[1] - got[0], mu[1] - mu[0],
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(race_probabilities(got, V=V, D=D, F=F, W=W),
                               p, rtol=1e-12, atol=1e-15)


def test_discrete_inverse_of_a_tiny_share():
    # the smaller share is solved in logs, so 1e-12 keeps its digits
    mu, V, D, F, W = list(_discrete_cases())[2]
    t = np.array([1e-12, 1.0 - 1e-12])
    got = abilities_from_race(t, V=V, D=D, F=F, W=W)
    p = race_probabilities(got, V=V, D=D, F=F, W=W)
    np.testing.assert_allclose(p[0], 1e-12, rtol=1e-9)


def test_default_rule_pairs_keep_the_closed_form():
    # no caller nodes: the closed form Phi((mu1 - mu0) / sd), bit for bit
    rng = np.random.default_rng(11)
    for _ in range(20):
        mu, D = rng.normal(size=2), rng.uniform(0.3, 2.0, size=2)
        V = rng.normal(size=(2, int(rng.integers(1, 4))))
        dV = V[0] - V[1]
        sd = np.sqrt(dV @ dV + D[0] + D[1])
        u = (mu[1] - mu[0]) / sd
        p = race_probabilities(mu, V=V, D=D)
        assert abs(p[0] - norm.cdf(u)) < 1e-15
        got = abilities_from_race(p, V=V, D=D)
        np.testing.assert_allclose(got[1] - got[0], mu[1] - mu[0],
                                   atol=1e-12)
