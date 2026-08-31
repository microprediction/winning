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


def test_fast_and_streaming_kernels_agree_exactly():
    """The hybrid seam: identical inputs through both memory layouts must be
    machine-identical (forced via the fast_max_entries threshold)."""
    fastrace = pytest.importorskip("fastrace")
    if not hasattr(fastrace, "block_race"):
        pytest.skip("fastrace without block_race")
    from scipy.special import roots_hermitenorm
    rng = np.random.default_rng(31)
    n, C = 4000, 80
    cl = np.sort(rng.integers(0, C, n))
    mu = rng.normal(0, 1.0, n)
    sd = 0.4 + 0.6 * rng.random(n)
    v = 0.7 * sd
    starts = np.flatnonzero(np.r_[True, np.diff(cl) != 0]).astype(np.int64)
    an, aw = roots_hermitenorm(9); aw = aw / aw.sum()
    args = (np.ascontiguousarray(mu), np.ascontiguousarray(sd),
            np.ascontiguousarray(v), starts, np.ascontiguousarray(an),
            np.ascontiguousarray(aw), 257)
    try:
        p_fast = np.asarray(fastrace.block_race(*args, np.nan, np.nan,
                                                10_000_000_000))
        p_stream = np.asarray(fastrace.block_race(*args, np.nan, np.nan, 0))
    except TypeError:
        pytest.skip("fastrace without fast_max_entries")
    assert np.abs(p_fast - p_stream).max() < 1e-14


def test_rank_r_blocks_forward_inverts_and_jacobian_refuses():
    """Rank-r cluster loadings: the forward kernel prices them (verified
    against Monte Carlo at 1e-3 TV when this was pinned), the inversion
    round-trips through the variance-matched preconditioner, and the
    Jacobian refuses cleanly rather than mis-broadcasting."""
    import warnings
    from winning.factor.races import race_probabilities, abilities_from_race
    from winning.factor.structures import Blocks
    from winning.factor.blocks import block_race_jacobian

    n = 24
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(n) * 0.6
    mu -= mu.mean()
    lab = np.repeat(np.arange(4), 6)
    D = 0.5 + 0.5 * rng.random(n)
    V2 = rng.standard_normal((n, 2)) * 0.5

    B = Blocks(cluster=lab, loading=V2, D=D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = race_probabilities(mu, structure=B)
        mu_back = abilities_from_race(p, structure=B)
        p2 = race_probabilities(mu_back, structure=B)
    assert 0.5 * np.abs(p2 - p).sum() < 1e-7

    with pytest.raises(NotImplementedError):
        block_race_jacobian(mu, lab, V2, D)


def test_sharp_blocks_warn_that_gh_cannot_converge():
    """A cluster loading of 4 against idiosyncratic sd 0.22 is sharpness
    18; the fixed-order hierarchical quadrature measured 5e-2 TV against
    a 4M-draw referee there, landing on GROUP shares, and raising the
    order does not converge (9 -> 31 nodes moves the answer by a further
    8e-2). Until the factor path's family escalation is ported, the
    kernel must say so."""
    import warnings
    from winning.factor.blocks import block_race_probabilities

    n = 24
    lab = np.repeat(np.arange(4), 6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        block_race_probabilities(np.zeros(n), lab, np.full(n, 4.0),
                                 np.full(n, 0.05))
        assert any("sharpness" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        block_race_probabilities(np.zeros(n), lab, np.full(n, 0.6),
                                 np.full(n, 0.7))
        assert not any("sharpness" in str(x.message) for x in w)


def test_window_covers_near_common_shock_winner_mass():
    """Fourteenth review's blocker: a 400-runner single cluster at
    correlation ~0.99, split into loadings 1.0 and 0.9 with equal mu.
    The old independent-marginal window lost 28 percent of the winner
    mass asymmetrically across the loading groups and the silent
    normalization returned group shares 0.68/0.32 where exchangeability
    within each group plus the symmetric construction forces the split
    to stay near 0.50/0.50. The node-aware window prices it exactly."""
    from winning.factor.blocks import block_race_probabilities

    n = 400
    p = block_race_probabilities(
        mu=np.zeros(n), cluster=np.zeros(n, dtype=int),
        loading=np.r_[np.ones(n // 2), 0.9 * np.ones(n // 2)],
        D=0.01 * np.ones(n), points=1025, qa=9)
    assert abs(p[: n // 2].sum() - 0.5) < 1e-4
    p2 = block_race_probabilities(
        mu=np.zeros(n), cluster=np.zeros(n, dtype=int),
        loading=np.r_[np.ones(n // 2), 0.9 * np.ones(n // 2)],
        D=0.01 * np.ones(n), points=257, qa=9)
    assert abs(p2[: n // 2].sum() - 0.5) < 1e-3


def test_mass_defect_raises_instead_of_normalizing():
    """A material raw-mass defect means the lattice missed part of the
    winner distribution; normalizing it away returns confident wrong
    shares, so the kernels must stop instead."""
    from winning.factor.blocks import _checked_mass

    ok = np.array([0.5, 0.499])
    out = _checked_mass(ok.copy(), "test race")
    assert abs(out.sum() - 1.0) < 1e-12
    with pytest.raises(RuntimeError, match="captured total mass"):
        _checked_mass(np.array([0.5, 0.22]), "test race")
    with pytest.raises(RuntimeError, match="captured total mass"):
        _checked_mass(np.array([0.9, 0.9]), "test race")


def test_zero_strength_tree_traversal_matches_independent():
    """Traversal-order regression: with all internal strengths zero a
    tree race IS an independent race, but ordering the message passes by
    the |strength| path sum (all tied at zero) visited children before
    their parents and priced a 6-leaf linkage tree at raw mass 3.0.
    Ordering by hop depth fixes it; both backends must agree."""
    import winning.factor.blocks as B
    from winning.factor.races import race_probabilities

    mu = np.array([0.3, 0.1, 0.0, -0.1, -0.2, 0.4])
    mu -= mu.mean()
    # depth-2 binary tree over six singleton leaf clusters, zero strengths
    parent = np.array([6, 8, 8, 7, 6, 7, 10, 9, 9, 10, -1])
    strength = np.zeros(11)
    D = np.ones(6)
    p_ind = race_probabilities(mu, D=D)
    for use_rust in ([True] if B._HAVE_RUST else []) + [False]:
        saved = B._HAVE_RUST
        B._HAVE_RUST = use_rust
        try:
            p = B.tree_race_probabilities(mu, np.arange(6), np.zeros(6),
                                          D, parent, strength)
        finally:
            B._HAVE_RUST = saved
        assert 0.5 * np.abs(p - p_ind).sum() < 1e-9


def test_from_linkage_tree_prices_cleanly():
    """End-to-end: a scipy linkage tree (the HRP entry point, whose
    floored merges produce exactly the zero strengths of the traversal
    regression) prices without a mass defect."""
    scipy_hier = pytest.importorskip("scipy.cluster.hierarchy")
    from winning.factor.structures import Tree
    from winning.factor.blocks import tree_race_probabilities

    rng = np.random.default_rng(5)
    Z = scipy_hier.linkage(rng.standard_normal((12, 4)), method="average")
    t = Tree.from_linkage(Z)
    mu = rng.normal(0, 0.8, 12)
    mu -= mu.mean()
    p = tree_race_probabilities(mu, t.cluster, t.loading, t.D,
                                t.parent, t.strength)
    assert abs(p.sum() - 1.0) < 1e-12
    assert (p > 0).all()


def test_advertised_surface_all_structures():
    """Every structure the front door advertises must price (and, where
    advertised, differentiate) without shape errors -- the rank-r block
    crash slipped through because nothing exercised the full grid."""
    import warnings
    from winning.factor.races import race_probabilities
    from winning.factor.structures import (Independent, Factor, Blocks,
                                           Nested, Tree)

    n = 12
    rng = np.random.default_rng(7)
    mu = rng.standard_normal(n) * 0.5
    mu -= mu.mean()
    D = 0.4 + 0.4 * rng.random(n)
    lab = np.repeat(np.arange(3), 4)
    v1 = 0.5 * np.ones(n)
    V2 = rng.standard_normal((n, 2)) * 0.4
    g = 0.3 * np.ones(n)
    parent = np.r_[np.full(3, 3), [-1]]
    structures = [
        Independent(D),
        Factor(V=v1.reshape(-1, 1), D=D),
        Factor(V=V2, D=D),
        Blocks(cluster=lab, loading=v1, D=D),
        Blocks(cluster=lab, loading=V2, D=D),
        Nested(cluster=lab, loading=v1, D=D, coupling=g),
        Tree(cluster=lab, loading=v1, D=D, parent=parent,
             strength=np.r_[np.zeros(3), 0.4]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for st in structures:
            p = race_probabilities(mu, structure=st)
            assert abs(p.sum() - 1.0) < 1e-9, type(st).__name__
            assert (p >= 0).all(), type(st).__name__
