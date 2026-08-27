"""Tests for winning.factor.polish: the race Jacobian and concentration
polishing (the core transport primitive; allocation etc. are thin users)."""
import numpy as np
import pytest

from winning.factor.polish import (race_jacobian, polish_race,
                                   concentration_matrix)
from winning.factor.races import race_probabilities


def _problem(seed=5, n=12):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    g = rng.normal(0, 0.6, n); g -= g.mean()
    D = 0.5 + 0.5 * rng.random(n)
    return mu, g[:, None], D


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_jacobian_matches_finite_differences(base):
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D, base=base)
    J = race_jacobian(mu, V=V, D=D, base=base)
    for j in range(len(mu)):
        m2 = mu.copy(); m2[j] += 1e-6
        fd = (race_probabilities(m2, V=V, D=D, base=base) - p0) / 1e-6
        assert np.abs(fd - J[:, j]).max() < 5e-6
    assert np.abs(J.sum(axis=1)).max() < 1e-12   # shift invariance


def test_polish_hits_caps_and_stays_a_race():
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D)
    big = int(np.argmax(p0))
    sector = [int(i) for i in np.argsort(-p0)[:4]]
    caps = np.full(len(mu), np.nan); caps[big] = 0.25
    p, mu1, info = polish_race(mu0=mu, V=V, D=D, name_caps=caps,
                               groups=[(sector, 0.60)])
    assert p[big] <= 0.25 + 1e-8
    assert p[sector].sum() <= 0.60 + 1e-8
    assert info["max_violation"] < 1e-8
    # the deliverable property: output IS the race at the returned abilities
    assert np.abs(p - race_probabilities(mu1, V=V, D=D)).max() < 1e-12
    # binding constraints are active, others untouched in direction
    assert len(info["active"]) >= 1


def test_polish_noop_when_caps_slack():
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D)
    p, mu1, info = polish_race(mu0=mu, V=V, D=D, name_caps=np.full(len(mu), 0.99))
    assert np.abs(p - p0).max() < 1e-6
    assert info["mu_distance"] < 1e-4


def test_polish_from_probabilities_entrypoint():
    mu, V, D = _problem(seed=7)
    p0 = race_probabilities(mu, V=V, D=D)
    p, mu1, info = polish_race(p0=p0, V=V, D=D,
                               name_caps=np.where(np.arange(len(mu)) == np.argmax(p0),
                                                  0.8 * p0.max(), np.nan))
    assert p.max() <= 0.8 * p0.max() + 1e-6
    assert abs(p.sum() - 1.0) < 1e-9


def test_concentration_matrix_shapes():
    A, b = concentration_matrix(5, name_caps=[0.3, np.nan, 0.2, np.nan, np.nan],
                                groups=[([0, 1], 0.5)])
    assert A.shape == (3, 5) and len(b) == 3
    assert b[-1] == 0.5 and A[-1, :2].sum() == 2.0


def test_hrp_polish_under_blocks_redistributes_along_the_dendrogram():
    """The HRP case: hierarchy-consistent covariance. Capping a name must
    push its mass DISPROPORTIONATELY to cluster siblings (they win in the
    same states), which independence-polishing cannot know."""
    from winning.factor.structures import Blocks, Independent
    rng = np.random.default_rng(17)
    n = 12
    cl = np.repeat(np.arange(3), 4)                # 3 sectors of 4
    mu = rng.normal(0, 0.6, n); mu -= mu.mean()
    D = (0.6 + 0.4 * rng.random(n)) ** 2
    v = 0.9 * np.sqrt(D)                           # strong within-sector corr
    S_blocks = Blocks(cl, v, D)
    from winning.factor.blocks import block_race_probabilities
    p0 = block_race_probabilities(mu, cl, v, D)
    big = int(np.argmax(p0))
    caps = np.full(n, np.nan); caps[big] = 0.7 * p0[big]
    p_b, mu_b, info_b = polish_race(mu0=mu, name_caps=caps,
                                    structure=S_blocks)
    assert p_b[big] <= caps[big] + 1e-7
    assert np.abs(p_b - block_race_probabilities(mu_b, cl, v, D)).max() < 1e-10
    # independence polish of the same weights for comparison
    p_i, _, _ = polish_race(p0=p0, name_caps=caps,
                            structure=Independent(D))
    sib = (cl == cl[big]) & (np.arange(n) != big)
    oth = cl != cl[big]
    gain_sib_b = (p_b[sib].sum() - p0[sib].sum())
    gain_sib_i = (p_i[sib].sum() - p0[sib].sum())
    # the freed mass is the same order; the block model gives siblings more
    assert gain_sib_b > gain_sib_i
    assert info_b["max_violation"] < 1e-7


def test_nested_structure_polish_runs_and_binds():
    from winning.factor.structures import Nested
    rng = np.random.default_rng(19)
    n = 10
    cl = np.repeat(np.arange(2), 5)
    mu = rng.normal(0, 0.5, n); mu -= mu.mean()
    D = (0.7 + 0.3 * rng.random(n)) ** 2
    v = 0.6 * np.sqrt(D)
    g = rng.normal(0, 0.4, n); g -= g.mean()
    S = Nested(cl, v, D, g, 1.0)
    from winning.factor.blocks import nested_race_probabilities
    p0 = nested_race_probabilities(mu, cl, v, D, coupling=g, gamma=1.0)
    big = int(np.argmax(p0))
    caps = np.full(n, np.nan); caps[big] = 0.75 * p0[big]
    p, mu1, info = polish_race(mu0=mu, name_caps=caps, structure=S)
    assert p[big] <= caps[big] + 1e-6
    assert info["max_violation"] < 1e-6


def test_tree_jacobian_flat_equals_block():
    """A depth-1 tree with zero strengths IS the block race; its Jacobian
    must reduce to the exact block Jacobian to machine precision."""
    from winning.factor.blocks import tree_race_jacobian, block_race_jacobian
    rng = np.random.default_rng(3)
    n = 12
    mu = rng.normal(size=n); D = 0.5 + rng.random(n)
    cluster = np.repeat(np.arange(4), 3); ld = 0.2 + 0.3 * rng.random(n)
    parent = np.array([4, 4, 4, 4, -1]); lam = np.zeros(5)
    Jt = tree_race_jacobian(mu, cluster, ld, D, parent, lam, points=257)
    Jb = block_race_jacobian(mu, cluster, ld, D, points=257)
    assert np.abs(Jt - Jb).max() < 1e-12
    assert np.abs(Jt.sum(axis=1)).max() < 1e-12


def test_polish_tree_from_linkage_enforces_caps():
    """polish_race(structure=Tree.from_linkage(Z)): polishing along the
    dendrogram (HRP's own belief) satisfies the caps exactly, via the
    finite-difference fallback where the analytic tree Jacobian's
    cross-cluster approximation would mislead SLSQP."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from winning.factor.structures import Tree
    rng = np.random.default_rng(3)
    n = 12
    blocks = np.repeat(np.arange(4), 3); superb = blocks // 2
    R = (0.15 + 0.25 * (superb[:, None] == superb[None, :])
         + 0.35 * (blocks[:, None] == blocks[None, :]))
    np.fill_diagonal(R, 1.0)
    Z = linkage(squareform(np.sqrt(0.5 * (1 - R)), checks=False),
                method="average")
    tree = Tree.from_linkage(Z)
    w0 = rng.dirichlet(np.ones(n) * 3)
    p, mu, info = polish_race(p0=w0, structure=tree, name_caps=0.14)
    assert p.max() <= 0.14 + 1e-6
    assert info["max_violation"] < 1e-6
    assert abs(p.sum() - 1.0) < 1e-9
