"""Anchor: factor gumbel == textbook mixed logit under Gauss-Hermite.

The correlated-softmax path (race_probabilities with V and
base="gumbel") has a machinery-free reference: conditional on the
Gaussian factors the race is exactly softmax, so all shares are
E_f[softmax(beta*(u + V f))] with beta = sqrt(pi^2/6) accounting for
the engine's unit-variance standardization of the Gumbel base. The
anchor shares no code with the lattice/factor engine.

Measured decomposition (pinned here): on the SAME factor grid the
engine matches the anchor to 8e-14 -- the lattice pass is the
mixed-logit expectation at machine precision -- and the residual in the
default-settings comparison (about 1e-4) is the difference between the
engine's sharpness-adaptive quadrature order and the anchor's fixed
32-node rule, not lattice error (points= does not move it).
"""

import numpy as np

from winning import calibrate_abilities, race_probabilities
from winning.factor.races import softmax_probabilities

BETA = np.sqrt(np.pi ** 2 / 6)  # unit-variance Gumbel standardization


def _gauss_hermite_grid(k, n=32):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    if k == 1:
        return x[:, None], w
    nodes = np.stack(np.meshgrid(*([x] * k), indexing="ij"), axis=-1).reshape(-1, k)
    wts = np.prod(np.stack(np.meshgrid(*([w] * k), indexing="ij"),
                           axis=-1).reshape(-1, k), axis=1)
    return nodes, wts


def _mixed_logit_shares(u, V, nodes=None, wts=None):
    if nodes is None:
        nodes, wts = _gauss_hermite_grid(V.shape[1])
    z = BETA * (u[None, :] + nodes @ V.T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return wts @ (e / e.sum(axis=1, keepdims=True))


def test_independent_gumbel_is_scaled_softmax():
    mu = np.array([-0.9, -0.1, 0.2, 0.5, 0.3])
    p = race_probabilities(mu, base="gumbel", points=4001)
    z = -BETA * mu
    expect = np.exp(z - z.max())
    expect = expect / expect.sum()
    assert np.abs(p - expect).max() < 1e-8


def test_factor_gumbel_matches_mixed_logit_on_the_same_grid():
    # identical factor nodes on both sides isolates the lattice pass,
    # which reproduces the anchor at machine precision (measured 8.4e-14)
    rng = np.random.default_rng(7)
    for _ in range(3):
        n = 6
        u = rng.normal(0, 1, n)
        V = rng.normal(0, 0.8, (n, 2))
        nodes, wts = _gauss_hermite_grid(2)
        p_engine = race_probabilities(-u, V=V, base="gumbel",
                                      F=nodes, W=wts, points=2001)
        p_anchor = _mixed_logit_shares(u, V, nodes, wts)
        assert np.abs(p_engine - p_anchor).max() < 1e-12


def test_factor_gumbel_default_quadrature_near_anchor():
    # default sharpness-adaptive order vs the anchor's fixed 32-node
    # rule: the gap is quadrature-order disagreement, ~1e-4 here
    rng = np.random.default_rng(7)
    for _ in range(3):
        n = 6
        u = rng.normal(0, 1, n)
        V = rng.normal(0, 0.8, (n, 2))
        p_engine = race_probabilities(-u, V=V, base="gumbel")
        p_anchor = _mixed_logit_shares(u, V)
        assert np.abs(p_engine - p_anchor).max() < 5e-4


def test_rank_one_covers_the_non_tensor_branch():
    rng = np.random.default_rng(11)
    u = rng.normal(0, 1, 6)
    V = rng.normal(0, 0.8, (6, 1))
    nodes, wts = _gauss_hermite_grid(1)
    p_engine = race_probabilities(-u, V=V, base="gumbel",
                                  F=nodes, W=wts, points=2001)
    assert np.abs(p_engine - _mixed_logit_shares(u, V, nodes, wts)).max() < 1e-9


def test_closed_form_mixture_equals_anchor():
    # the packaged softmax_probabilities at tau = 1/beta is the same
    # mixture the anchor computes, through the package's own code path
    rng = np.random.default_rng(3)
    u = rng.normal(0, 1, 7)
    V = rng.normal(0, 0.8, (7, 2))
    nodes, wts = _gauss_hermite_grid(2)
    p_pkg = softmax_probabilities(-u, temperature=1.0 / BETA, V=V,
                                  F=nodes, W=wts)
    assert np.abs(p_pkg - _mixed_logit_shares(u, V, nodes, wts)).max() < 1e-14


def test_factor_gumbel_calibration_roundtrips_through_anchor():
    u = np.array([0.9, 0.1, -0.2, -0.5, -0.3])
    V = np.array([[1.5, 0], [1.5, 0], [0, 1.5], [0, 1.5], [0, 0]], dtype=float)
    p = _mixed_logit_shares(u, V)
    mu_hat = np.asarray(calibrate_abilities(p, V=V, base="gumbel"))
    u_hat = -(mu_hat - mu_hat.mean())
    assert np.abs(u_hat - (u - u.mean())).max() < 1e-3


def test_berry_contraction_agrees_on_the_inverse():
    # Berry (1994): delta <- delta + (log s_obs - log s(delta)) converges
    # for logit-family shares; run it on the anchor's own forward map and
    # compare with the engine's inversion. Two inversions, no shared code.
    rng = np.random.default_rng(5)
    u = rng.normal(0, 0.8, 6)
    u -= u.mean()
    V = rng.normal(0, 0.7, (6, 2))
    nodes, wts = _gauss_hermite_grid(2)
    s_obs = _mixed_logit_shares(u, V, nodes, wts)
    delta = np.zeros(6)
    for _ in range(400):
        s = _mixed_logit_shares(delta / BETA, V, nodes, wts)
        step = np.log(s_obs) - np.log(s)
        delta = delta + step
        if np.abs(step).max() < 1e-13:
            break
    u_berry = delta / BETA
    u_berry -= u_berry.mean()
    assert np.abs(u_berry - u).max() < 1e-10
    mu_hat = np.asarray(calibrate_abilities(s_obs, V=V, base="gumbel"))
    u_engine = -(mu_hat - mu_hat.mean())
    assert np.abs(u_engine - u_berry).max() < 1e-3
