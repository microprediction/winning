"""Top-k memberships at factor rank above two, and under the caller's own
factor law F, W (#623).

The node rule is the general race's -- deterministic scrambled Sobol at
these ranks, never Monte Carlo -- so a covariance from fit_covariance
(k = 3 global factors plus block columns, with its own 2048 nodes)
prices shortlists directly. Referees are fixed-seed Monte Carlo draws of
the SAME Gaussian, at tolerances several times their noise.
"""
import numpy as np
import pytest

from winning.factor.core import fit_covariance, hermite_nodes, qmc_nodes
from winning.factor.races import race_probabilities
from winning.factor.topk import (abilities_from_topk, bottom_k_probabilities,
                                 rank_probabilities, top_k_jacobians,
                                 top_k_probabilities)


def _mc_topk(mu, Sig, k, draws=400_000, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(len(mu)))
    X = mu + rng.standard_normal((draws, len(mu))) @ L.T
    th = np.partition(X, k - 1, axis=1)[:, k - 1]
    return (X <= th[:, None]).mean(axis=0)


def _rank3_field(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.5, n), rng.uniform(0.4, 1.0, n),
            rng.normal(0, 0.5, (n, 3)))


def test_fit_covariance_with_labels_to_top_k_end_to_end():
    """fit_covariance(C, k=3, blocks=labels) -> top_k_probabilities, the
    whole intake, against a fixed-seed Monte Carlo of the FITTED Gaussian
    (V V' + diag(D)). 4e5 draws: per-runner sd <= 8e-4; measured max
    error 2.4e-3 with the fit's own 2048 nodes. (A fit whose rank nears
    n can drive D to its floor -- the near-deterministic regime, out of
    scope -- so the field is sized to keep D clear of it.)"""
    rng = np.random.default_rng(3)
    n_fam, size = 6, 4
    n = n_fam * size
    fam = np.repeat(np.arange(n_fam), size)
    G = rng.normal(0, 0.6, (n, 3))
    B = np.zeros((n, n_fam))
    B[np.arange(n), fam] = rng.uniform(0.4, 0.8, n)
    C = G @ G.T + B @ B.T + np.diag(rng.uniform(0.2, 0.5, n))
    V, D, F, W, rep = fit_covariance(C, k=3, blocks=fam, return_report=True)
    assert rep["blocks"] == "given" and V.shape[1] > 3
    assert D.min() > 0.01
    mu = rng.normal(0, 0.5, n)
    q = top_k_probabilities(mu, 3, V, D, F=F, W=W)
    assert abs(q.sum() - 3) < 1e-9
    emp = _mc_topk(mu, V @ V.T + np.diag(D), 3, seed=7)
    assert np.abs(q - emp).max() < 6e-3
    # the default rule (no F, W) prices the same law
    assert np.abs(top_k_probabilities(mu, 3, V, D) - emp).max() < 6e-3
    # k = 1 under the fit's nodes IS race_probabilities under them
    assert np.allclose(top_k_probabilities(mu, 1, V, D, F=F, W=W),
                       race_probabilities(mu, V, D, F, W), atol=2e-6)


def test_rank_three_against_monte_carlo():
    mu, D, V = _rank3_field()
    q = top_k_probabilities(mu, 3, V=V, D=D)
    emp = _mc_topk(mu, V @ V.T + np.diag(D), 3, seed=11)
    assert np.abs(q - emp).max() < 4e-3          # measured 9e-4 at 1e6


def test_rank_three_invariances():
    """Basis rotation V -> VQ (same covariance) moves only the node
    error -- measured 8.5e-4 on 1024 Sobol nodes; permuting entrants
    permutes the answer exactly; a common loading column is gauge."""
    mu, D, V = _rank3_field()
    q = top_k_probabilities(mu, 3, V=V, D=D)
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    assert np.abs(top_k_probabilities(mu, 3, V=V @ Q, D=D) - q).max() < 3e-3
    perm = np.random.default_rng(1).permutation(len(mu))
    qp = top_k_probabilities(mu[perm], 3, V=V[perm], D=D[perm])
    assert np.allclose(qp, q[perm], atol=1e-12)
    qg = top_k_probabilities(mu, 3, V=V + np.array([1.0, -2.0, 0.5]), D=D)
    assert np.allclose(qg, q, atol=1e-12)


def test_explicit_qa_at_rank_three_is_the_pruned_product_rule():
    mu, D, V = _rank3_field()
    q9 = top_k_probabilities(mu, 3, V=V, D=D, qa=9)
    Fh, Wh = hermite_nodes(3, Q=9)
    assert np.allclose(top_k_probabilities(mu, 3, V=V, D=D, F=Fh, W=Wh),
                       q9, atol=1e-14)
    assert np.abs(q9 - top_k_probabilities(mu, 3, V=V, D=D)).max() < 2e-3


def test_caller_nodes_are_the_factor_law():
    """F, W stand for the factor law as race_probabilities reads them:
    relative weights, and a one-node law is a deterministic shift."""
    mu, D, V = _rank3_field()
    F, W = qmc_nodes(3, m=9)
    q = top_k_probabilities(mu, 2, V=V, D=D, F=F, W=W)
    assert np.allclose(top_k_probabilities(mu, 2, V=V, D=D, F=F, W=7 * W),
                       q, atol=1e-14)
    f0 = np.array([[0.3, -1.0, 0.2]])
    one = top_k_probabilities(mu, 2, V=V, D=D, F=f0, W=[1.0])
    shifted = top_k_probabilities(mu + (V - V.mean(0)) @ f0[0], 2, D=D)
    assert np.allclose(one, shifted, atol=1e-12)


def test_siblings_extend_to_rank_three():
    """bottom_k, rank_probabilities, top_k_jacobians and
    abilities_from_topk share the node helper, so they take rank three
    too; each agrees with top_k_probabilities where they must."""
    mu, D, V = _rank3_field(n=6)
    q = top_k_probabilities(mu, 2, V=V, D=D)
    P = rank_probabilities(mu, D=D, V=V)
    assert np.abs(P[:, :2].sum(axis=1) - q).max() < 1e-7
    # symmetric base: the k largest of X are the k smallest of -X
    assert np.allclose(bottom_k_probabilities(mu, 2, V=V, D=D),
                       top_k_probabilities(-mu, 2, V=-V, D=D), atol=1e-12)
    # the derivative and the inverse on a cheap explicit rule (qa=5, the
    # pruned 5^3 product): the node helper is the same either way
    Jm, _ = top_k_jacobians(mu, 2, D=D, V=V, qa=5)
    h = 1e-5
    e = np.zeros(len(mu))
    e[1] = h
    fd = (top_k_probabilities(mu + e, 2, V=V, D=D, qa=5)
          - top_k_probabilities(mu - e, 2, V=V, D=D, qa=5)) / (2 * h)
    assert np.abs(fd - Jm[:, 1]).max() < 1e-7
    q = top_k_probabilities(mu, 2, V=V, D=D, qa=5)
    m2 = abilities_from_topk(q, 2, V=V, D=D, qa=5)
    assert np.abs((m2 - m2.mean()) - (mu - mu.mean())).max() < 1e-6


def test_rule_validation():
    mu, D, V = _rank3_field()
    F, W = qmc_nodes(3, m=6)
    with pytest.raises(ValueError, match="both F"):
        top_k_probabilities(mu, 2, V=V, D=D, F=F)
    with pytest.raises(ValueError, match="both F"):
        top_k_probabilities(mu, 2, V=V, D=D, W=W)
    with pytest.raises(ValueError, match="factor columns"):
        top_k_probabilities(mu, 2, V=V[:, :2], D=D, F=F, W=W)
    with pytest.raises(ValueError, match="one row per"):
        top_k_probabilities(mu, 2, V=V, D=D, F=F, W=W[:-1])
    with pytest.raises(ValueError, match="need loadings V"):
        top_k_probabilities(mu, 2, D=D, F=F, W=W)
    with pytest.raises(ValueError, match="qa="):
        top_k_probabilities(mu, 2, V=V, D=D, F=F, W=W, qa=7)
    with pytest.raises(ValueError):
        top_k_probabilities(mu, 2, V=V, D=D, F=F, W=-W)


def test_failed_lattice_is_refined_within_a_cap(monkeypatch):
    """A lattice that fails the slot check is doubled, at most to four
    times the request; a defect that survives raises naming points=."""
    import winning.factor.topk as tk
    real = tk._topk_independent
    seen = []

    def coarse_below(limit):
        def fake(mu, sd, k, base_rows, points, **kw):
            seen.append(points)
            raw = real(mu, sd, k, base_rows, points, **kw)
            return raw * (1.5 if points < limit else 1.0)
        return fake

    mu = np.array([0.0, 0.3, -0.2, 0.5, 0.1])
    ref = top_k_probabilities(mu, 2)
    monkeypatch.setattr(tk, "_topk_independent", coarse_below(2049))
    q = top_k_probabilities(mu, 2)
    assert seen == [513, 1025, 2049]
    assert np.allclose(q, top_k_probabilities(mu, 2, points=2049))
    assert np.allclose(q, ref, atol=1e-9)
    seen.clear()
    monkeypatch.setattr(tk, "_topk_independent", coarse_below(10 ** 6))
    with pytest.raises(RuntimeError, match=r"points=513 to 2049"):
        top_k_probabilities(mu, 2)
    assert seen == [513, 1025, 2049]
    seen.clear()
    with pytest.raises(RuntimeError, match="slots"):
        top_k_probabilities(mu, 2, points=8193)   # at the cap: no retry
    assert seen == [8193]
