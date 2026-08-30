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


def test_cophenetic_tree_race_identity():
    """HRP's implicit covariance IS a tree race, exactly: build the tree from
    the linkage with lam^2 = cophenetic correlation increment per merge, and
    the implied correlation equals the cophenetic matrix to the last bit.
    (Increments nonnegative by linkage monotonicity.)"""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    n = 12
    blocks = np.repeat(np.arange(4), 3); superb = blocks // 2
    R = (0.15 * np.ones((n, n)) + 0.25 * (superb[:, None] == superb[None, :])
         + 0.35 * (blocks[:, None] == blocks[None, :]))
    np.fill_diagonal(R, 1.0)
    d = np.sqrt(0.5 * (1.0 - R))
    Z = linkage(squareform(d, checks=False), method="average")
    coph = 1.0 - 2.0 * squareform(cophenet(Z)) ** 2
    np.fill_diagonal(coph, 1.0)
    parent = -np.ones(2 * n - 1, int)
    rho = np.zeros(2 * n - 1)
    for k, (a, b, h, _) in enumerate(Z):
        t = n + k
        parent[int(a)] = t; parent[int(b)] = t
        rho[t] = 1.0 - 2.0 * h * h
    lam2 = np.zeros(2 * n - 1)
    for t in range(n, 2 * n - 1):
        pa = parent[t]
        lam2[t] = rho[t] - (rho[pa] if pa >= 0 else 0.0)
    assert (lam2[n:] >= -1e-12).all()
    implied = np.eye(n)
    for i in range(n):
        anc_i = set(); u = i
        while parent[u] >= 0:
            anc_i.add(parent[u]); u = parent[u]
        for j in range(n):
            if i == j:
                continue
            anc_j = set(); u = j
            while parent[u] >= 0:
                anc_j.add(parent[u]); u = parent[u]
            implied[i, j] = sum(lam2[t] for t in anc_i & anc_j)
    assert np.abs(implied - coph).max() < 1e-14


def test_from_linkage_floors_negative_cophenetic_correlation():
    """Merges above the h = 1/sqrt(2) horizon imply NEGATIVE cophenetic
    correlation, which the tree race cannot represent; from_linkage floors
    rho at zero. Without the floor, clipping the negative root increment
    silently inflated implied correlations above one (caught by the
    general-Sigma control-variate experiment)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from winning.factor.structures import Tree
    rng = np.random.default_rng(11)
    n = 20
    B = rng.normal(size=(n, 2)) * [0.55, 0.3]
    S = B @ B.T + 0.5 * np.eye(n)
    d_ = np.sqrt(np.diag(S)); C = S / np.outer(d_, d_)
    Z = linkage(squareform(np.sqrt(np.clip(0.5 * (1 - C), 0, 1)),
                           checks=False), method="average")
    assert (1.0 - 2.0 * Z[-1, 2] ** 2) < 0    # the premise: negative at root
    tr = Tree.from_linkage(Z)
    lam2 = np.asarray(tr.strength) ** 2
    parent = np.asarray(tr.parent)
    implied = np.eye(n)
    for i in range(n):
        anc_i = set(); u = i
        while parent[u] >= 0:
            anc_i.add(parent[u]); u = parent[u]
        for j in range(n):
            if j == i:
                continue
            anc_j = set(); u = j
            while parent[u] >= 0:
                anc_j.add(parent[u]); u = parent[u]
            implied[i, j] = sum(lam2[t] for t in anc_i & anc_j)
    assert implied.max() <= 1.0 + 1e-12
    assert implied.min() >= -1e-12
    assert np.linalg.eigvalsh(implied).min() > -1e-10


def test_heavy_favorite_inversion_converges():
    # the dominance-window stall: a near-certain winner's residual and
    # own-slope both vanish, and their noisy ratio destabilized the
    # damped fixed point through the recentering (stalled at 0.31 max
    # log-residual for targets in the 1e-4..1e-8 window, any n >= 4;
    # the residual-proportional step cap converges these in 4-6
    # iterations). Round trips must hold across the whole window.
    import numpy as np
    from winning import race_probabilities
    from winning.factor.races import abilities_from_race
    for n in (4, 8, 20):
        for tiny in (1e-4, 1e-6, 1e-8, 1e-10):
            p = np.full(n, tiny)
            p[0] = 1 - (n - 1) * tiny
            mu = abilities_from_race(p / p.sum())
            p2 = race_probabilities(mu)
            assert np.abs(np.log(p2) - np.log(p / p.sum())).max() < 1e-6


def test_removal_shares_dominant_favorite():
    # fourth review: removing a contestant 20-200 sd better than the
    # field moves the post-removal winner bulk far from the original
    # one; the lattice must cover it and the row mass must be checked
    # rather than silently renormalized
    import numpy as np
    from scipy.stats import norm
    from winning.factor.races import removal_shares
    truth = norm.cdf(1.0 / np.sqrt(2.0))
    for gap in (20.0, 200.0):
        mu = np.array([-gap, 0.0, 1.0])
        q = removal_shares(mu, D=np.ones(3))
        assert abs(q[0, 1] - truth) < 1e-4
        assert abs(q.sum(axis=1) - 1).max() < 1e-9


def test_lattice_convergence_is_spectral_not_quadratic():
    # Peter's challenge ("we are on a lattice, there are always ties"):
    # within-cell tie mass lives in the quadrature error, and for these
    # smooth, tail-decaying integrands that error is SPECTRAL, not
    # O(dx^2) -- measured 5.6e-3 at 9 points, 7.9e-7 at 17, machine zero
    # by 33 on an asymmetric pair. This is why the factor engine needs
    # no multiplicity bookkeeping where the classic integer-lattice
    # engine (whose dead-heat mass is fixed and real) does.
    import numpy as np
    from scipy.stats import norm
    from winning.factor.races import race_probabilities
    mu = np.array([0.0, 0.3])
    exact = norm.cdf(0.3 / np.sqrt(2))
    p17 = race_probabilities(mu, D=np.ones(2), points=17, window="span")
    p33 = race_probabilities(mu, D=np.ones(2), points=33, window="span")
    assert abs(p17[0] - exact) < 1e-5
    assert abs(p33[0] - exact) < 1e-12
