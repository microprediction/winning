"""Invariant, anchor and fuzz tests for the race kernels.

Each test targets a CLASS of bug this codebase has actually produced:
permutation/segment juggling, sign conventions, lattice windows, floors,
common-mode leaks. Exact identities wherever possible; Monte Carlo nowhere.
"""
import numpy as np
import pytest
from scipy.stats import norm

from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   block_race_jacobian)
from winning.factor.races import race_probabilities


def _rand_structure(rng, n=None):
    n = n or int(rng.integers(5, 60))
    C = int(rng.integers(1, max(2, n // 2)))
    cl = rng.integers(0, C, n)
    mu = rng.normal(0, float(rng.uniform(0.2, 2.0)), n); mu -= mu.mean()
    # adversarial scales: up to 30:1 sd ratios
    D = np.exp(rng.uniform(-1.5, 1.5, n)) ** 2
    v = rng.uniform(0, 1.2, n) * np.sqrt(D)
    return mu, cl, v, D


# ---------------------------------------------------------------- anchors
def test_two_runner_closed_form_same_cluster():
    # same cluster: shared effect a with loadings v1, v2; difference is
    # Gaussian with var s1^2 + s2^2 + (v1 - v2)^2  (min-wins: p1 = P(Y1 < Y2))
    mu = np.array([0.3, -0.3]); D = np.array([0.7, 1.3]); v = np.array([0.9, 0.4])
    p = block_race_probabilities(mu, np.array([0, 0]), v, D, points=1001)
    s = np.sqrt(D.sum() + (v[0] - v[1]) ** 2)
    exact = norm.cdf((mu[1] - mu[0]) / s)
    assert abs(p[0] - exact) < 1e-6


def test_two_runner_closed_form_cross_cluster():
    # different clusters: independent effects; var s1^2 + s2^2 + v1^2 + v2^2
    mu = np.array([0.3, -0.3]); D = np.array([0.7, 1.3]); v = np.array([0.9, 0.4])
    p = block_race_probabilities(mu, np.array([0, 1]), v, D, points=1001)
    s = np.sqrt(D.sum() + (v ** 2).sum())
    exact = norm.cdf((mu[1] - mu[0]) / s)
    assert abs(p[0] - exact) < 1e-6


def test_exchangeable_runners_get_equal_probabilities():
    n = 12
    mu = np.zeros(n); D = np.full(n, 0.8); v = np.full(n, 0.5)
    cl = np.repeat(np.arange(4), 3)
    p = block_race_probabilities(mu, cl, v, D)
    assert np.abs(p - 1.0 / n).max() < 1e-9


# ------------------------------------------------------------- invariants
@pytest.mark.parametrize("seed", range(8))
def test_permutation_equivariance(seed):
    rng = np.random.default_rng(seed)
    mu, cl, v, D = _rand_structure(rng)
    p = block_race_probabilities(mu, cl, v, D)
    perm = rng.permutation(len(mu))
    p2 = block_race_probabilities(mu[perm], cl[perm], v[perm], D[perm])
    assert np.abs(p2 - p[perm]).max() < 1e-10


@pytest.mark.parametrize("seed", range(8))
def test_cluster_relabeling_invariance(seed):
    rng = np.random.default_rng(seed)
    mu, cl, v, D = _rand_structure(rng)
    p = block_race_probabilities(mu, cl, v, D)
    relabel = rng.permutation(cl.max() + 1)
    p2 = block_race_probabilities(mu, relabel[cl], v, D)
    assert np.abs(p2 - p).max() < 1e-10


@pytest.mark.parametrize("seed", range(4))
def test_scale_invariance(seed):
    rng = np.random.default_rng(seed)
    mu, cl, v, D = _rand_structure(rng)
    p = block_race_probabilities(mu, cl, v, D)
    c = 3.7
    p2 = block_race_probabilities(c * mu, cl, c * v, c * c * D)
    assert np.abs(p2 - p).max() < 1e-8


@pytest.mark.parametrize("seed", range(4))
def test_common_shift_invariance_fuzz(seed):
    rng = np.random.default_rng(100 + seed)
    mu, cl, v, D = _rand_structure(rng)
    p = block_race_probabilities(mu, cl, v, D)
    p2 = block_race_probabilities(mu + float(rng.uniform(-5, 5)), cl, v, D)
    assert np.abs(p2 - p).max() < 1e-8


def test_tree_common_root_is_invisible():
    rng = np.random.default_rng(11)
    mu, cl, v, D = _rand_structure(rng, n=40)
    nC = cl.max() + 1
    parent = np.array([nC] * nC + [-1])
    p0 = tree_race_probabilities(mu, cl, v, D, parent, np.zeros(nC + 1))
    p1 = tree_race_probabilities(mu, cl, v, D, parent,
                                 np.r_[np.zeros(nC), 2.0], points=513)
    assert 0.5 * np.abs(p0 - p1).sum() < 5e-4


# -------------------------------------------------- derivative identities
def test_jacobian_offdiagonal_symmetry_and_signs():
    rng = np.random.default_rng(3)
    mu, cl, v, D = _rand_structure(rng, n=20)
    J = block_race_jacobian(mu, cl, v, D)
    off = J - np.diag(np.diag(J))
    assert np.abs(off - off.T).max() < 1e-9          # dp_i/dmu_j = dp_j/dmu_i
    assert (np.diag(J) < 0).all()                    # min-wins: worse mu_i, lower p_i
    assert (off >= -1e-12).all()                     # others benefit


# --------------------------------------------------------------- fuzzing
@pytest.mark.parametrize("seed", range(12))
def test_fuzz_probabilities_are_sane(seed):
    rng = np.random.default_rng(1000 + seed)
    mu, cl, v, D = _rand_structure(rng)
    # occasionally make one runner dominant and one hopeless
    if seed % 3 == 0:
        mu[0] -= 8.0
        mu[-1] += 8.0
        mu -= mu.mean()
    p = block_race_probabilities(mu, cl, v, D)
    assert np.all(p >= 0) and abs(p.sum() - 1.0) < 1e-9
    assert np.all(np.isfinite(p))
    if seed % 3 == 0:
        assert p[0] == p.max()


def test_cross_engine_block_c1_equals_factor_race():
    # one big block with loadings == rank-1 factor race through the OTHER engine
    rng = np.random.default_rng(9)
    n = 30
    mu = rng.normal(0, 1, n); mu -= mu.mean()
    D = np.exp(rng.uniform(-1, 1, n)) ** 2
    v = rng.normal(0, 0.8, n)
    p_block = block_race_probabilities(mu, np.zeros(n, int), v, D, points=1001)
    p_factor = race_probabilities(mu, V=v[:, None], D=D, points=1001)
    assert 0.5 * np.abs(p_block - p_factor).sum() < 2e-3
