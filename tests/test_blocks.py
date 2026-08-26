"""winning.factor.blocks: package-grade checks (full validation history in
research/pqrace/SCHUR.md)."""
import numpy as np
import pytest

from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   block_race_jacobian,
                                   abilities_from_block_race)


def _prob(seed=3, n=60, c=10):
    rng = np.random.default_rng(seed)
    cl = rng.integers(0, c, n)
    mu = rng.normal(0, 1.0, n); mu -= mu.mean()
    D = (0.4 + 0.6 * rng.random(n)) ** 2
    v = 0.8 * np.sqrt(D)
    return mu, cl, v, D


def test_block_matches_monte_carlo():
    mu, cl, v, D = _prob()
    p = block_race_probabilities(mu, cl, v, D)
    rng = np.random.default_rng(0)
    M, n, C = 400_000, len(mu), cl.max() + 1
    cnt = np.zeros(n)
    for a in range(0, M, 50_000):
        m = min(50_000, M - a)
        A = rng.standard_normal((m, C))
        Y = -mu + v * A[:, cl] + np.sqrt(D) * rng.standard_normal((m, n))
        np.add.at(cnt, np.argmax(Y, axis=1), 1.0)     # min-wins = argmax of -Y+..
    mc = cnt / M
    assert 0.5 * np.abs(p - mc).sum() < 0.01


def test_degenerate_clusterings_agree():
    mu, cl, v, D = _prob()
    p1 = block_race_probabilities(mu, np.zeros(len(mu), int), np.zeros(len(mu)), D)
    p2 = block_race_probabilities(mu, np.arange(len(mu)), np.zeros(len(mu)), D)
    assert np.abs(p1 - p2).max() < 1e-12


def test_common_shift_invariance():
    mu, cl, v, D = _prob()
    p1 = block_race_probabilities(mu, cl, v, D)
    p2 = block_race_probabilities(mu + 3.7, cl, v, D)
    assert np.abs(p1 - p2).max() < 1e-8


def test_jacobian_matches_fd_and_sums_to_zero():
    mu, cl, v, D = _prob(n=25, c=6)
    p0 = block_race_probabilities(mu, cl, v, D)
    J = block_race_jacobian(mu, cl, v, D)
    assert np.abs(J.sum(axis=1)).max() < 1e-10
    for j in range(0, 25, 5):
        m2 = mu.copy(); m2[j] += 1e-6
        fd = (block_race_probabilities(m2, cl, v, D) - p0) / 1e-6
        assert np.abs(fd - J[:, j]).max() < 5e-5


def test_inversion_round_trip():
    mu, cl, v, D = _prob(n=40, c=8)
    p = block_race_probabilities(mu, cl, v, D)
    mu_hat, resid, _ = abilities_from_block_race(p, cl, v, D)
    assert resid < 1e-9
    assert np.abs(mu_hat - mu).max() < 1e-7


def test_nested_gamma_zero_is_block():
    mu, cl, v, D = _prob()
    g = np.linspace(-1, 1, len(mu))
    p0 = block_race_probabilities(mu, cl, v, D)
    p1 = nested_race_probabilities(mu, cl, v, D, coupling=g, gamma=0.0)
    p2 = nested_race_probabilities(mu, cl, v, D, coupling=g, gamma=1.0, qf=9)
    assert np.abs(p0 - p1).max() < 1e-12
    assert 0.5 * np.abs(p0 - p2).sum() > 0.01     # coupling does something


def test_one_race_five_grammars():
    from winning.factor.structures import (Independent, Factor, Blocks,
                                           Nested, Tree)
    from winning.factor.races import race_probabilities
    from winning.factor.blocks import tree_race_probabilities
    mu, cl, v, D = _prob(n=40, c=8)
    g = np.linspace(-0.5, 0.5, 40)
    # each structure through the ONE front door equals its direct kernel
    p_b = race_probabilities(mu, structure=Blocks(cl, v, D))
    assert np.abs(p_b - block_race_probabilities(mu, cl, v, D)).max() < 1e-12
    p_n = race_probabilities(mu, structure=Nested(cl, v, D, g, 1.0))
    assert abs(p_n.sum() - 1) < 1e-9
    p_i = race_probabilities(mu, structure=Independent(D))
    p_i2 = race_probabilities(mu, structure=Blocks(cl, np.zeros(40), D),)
    assert 0.5 * np.abs(p_i - p_i2).sum() < 2e-3   # containment (different engines)
    # tree: depth-1 tree with zero strengths reduces to blocks
    nC = cl.max() + 1
    parent = np.array([nC] * nC + [-1]); strength = np.zeros(nC + 1)
    p_t = race_probabilities(mu, structure=Tree(cl, v, D, parent, strength))
    assert 0.5 * np.abs(p_t - p_b).sum() < 2e-3


def test_rank2_blocks_with_rank2_coupling():
    """The US/Europe spec: 2 global factors (free loadings) over two regions,
    each region with its own 2 local factors (free loadings)."""
    rng = np.random.default_rng(7)
    n = 40
    region = (np.arange(n) >= n // 2).astype(int)
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    D = (0.5 + 0.5 * rng.random(n)) ** 2
    G = rng.normal(0, 0.5, (n, 2))
    Vl = rng.normal(0, 0.6, (n, 2))
    p = nested_race_probabilities(mu, region, Vl, D, coupling=G, gamma=1.0,
                                  qa=11, qf=8)
    assert abs(p.sum() - 1.0) < 1e-9
    M, cnt = 300_000, np.zeros(n)
    for a in range(0, M, 100_000):
        m = min(100_000, M - a)
        f = rng.standard_normal((m, 2))
        aU = rng.standard_normal((m, 2)); aE = rng.standard_normal((m, 2))
        loc = np.where(region[None, :, None] == 0, aU[:, None, :], aE[:, None, :])
        Y = -mu + f @ G.T + np.einsum("mnr,nr->mn", loc, Vl) \
            + np.sqrt(D) * rng.standard_normal((m, n))
        np.add.at(cnt, np.argmax(Y, axis=1), 1.0)
    assert 0.5 * np.abs(p - cnt / M).sum() < 0.012


def test_rank1_column_matrix_matches_vector_loading():
    mu, cl, v, D = _prob(n=30, c=6)
    p1 = block_race_probabilities(mu, cl, v, D)
    p2 = block_race_probabilities(mu, cl, v.reshape(-1, 1), D)
    assert np.abs(p1 - p2).max() < 1e-12
