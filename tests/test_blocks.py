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


def test_tree_refuses_rank_r_leaf_loadings():
    """The paper's tree grammar takes SCALAR leaf-cluster loadings (only
    the block grammar prices rank r). Python mis-broadcast cryptically
    and the R port silently priced the flattened matrix, so all tree
    entry points must refuse a matrix loading cleanly; an (n, 1) column
    is accepted as the scalar it is."""
    from winning.factor.blocks import (tree_race_probabilities,
                                       tree_race_jacobian)
    from winning.factor.races import race_probabilities
    from winning.factor.structures import Tree

    n = 12
    rng = np.random.default_rng(7)
    mu = rng.standard_normal(n) * 0.5
    mu -= mu.mean()
    D = 0.4 + 0.4 * rng.random(n)
    lab = np.repeat(np.arange(3), 4)
    V2 = rng.standard_normal((n, 2)) * 0.4
    parent = np.r_[np.full(3, 3), [-1]]
    strength = np.r_[np.zeros(3), 0.4]

    with pytest.raises(NotImplementedError, match="rank-one"):
        tree_race_probabilities(mu, lab, V2, D, parent, strength)
    with pytest.raises(NotImplementedError, match="rank-one"):
        tree_race_jacobian(mu, lab, V2, D, parent, strength)
    with pytest.raises(NotImplementedError, match="rank-one"):
        race_probabilities(mu, structure=Tree(cluster=lab, loading=V2, D=D,
                                              parent=parent,
                                              strength=strength))
    v1col = 0.5 * np.ones((n, 1))
    p_col = tree_race_probabilities(mu, lab, v1col, D, parent, strength)
    p_flat = tree_race_probabilities(mu, lab, v1col.ravel(), D, parent,
                                     strength)
    assert np.array_equal(p_col, p_flat)


def test_mass_check_rejects_nan():
    """NaN compares false against any tolerance, so without an explicit
    finiteness test a NaN mass sails through the defect check."""
    from winning.factor.blocks import _checked_mass

    with pytest.raises(RuntimeError, match="captured total mass"):
        _checked_mass(np.array([0.5, np.nan]), "test race")


@pytest.mark.parametrize("c", [1e100, 1e307, 1e308])
def test_nested_reciprocal_coupling_scaling_is_the_same_race_722(c):
    """coupling -> c coupling, gamma -> gamma / c leaves gamma g, hence
    the race, unchanged. At 1e308 g @ node overflowed before gamma (NaN
    lattice mass); at 1e307 the inverse's variance squared gamma and g
    separately, 0 * inf."""
    import warnings
    from winning.factor.structures import Nested
    from winning.factor.races import race_probabilities, abilities_from_race
    from winning.factor.blocks import nested_race_jacobian
    mu = np.array([-0.4, -0.1, 0.2, 0.5, 0.8])
    cl, ld = [0, 0, 1, 1, 2], [0.4, 0.2, 0.3, 0.5, 0.1]
    D = [0.7, 0.8, 0.9, 1.0, 1.1]
    g = np.array([-0.4, 0.1, 0.4, -0.2, 0.1])
    p1 = race_probabilities(mu, structure=Nested(cl, ld, D, g, 0.8))
    a1 = abilities_from_race(p1, structure=Nested(cl, ld, D, g, 0.8))
    J1 = nested_race_jacobian(mu, cl, ld, D, coupling=g, gamma=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = Nested(cl, ld, D, g * c, 0.8 / c)
        p = race_probabilities(mu, structure=s)
        a = abilities_from_race(p1, structure=s)
        J = nested_race_jacobian(mu, cl, ld, D, coupling=g * c, gamma=0.8 / c)
    assert np.abs(p - p1).max() < 1e-14
    assert np.abs(a - a1).max() < 1e-12
    assert np.isfinite(J).all() and np.abs(J - J1).max() < 1e-14


@pytest.mark.parametrize("c", [40.0, 50.0, 1e6])
def test_single_cluster_common_loading_is_a_gauge_471(c):
    """One cluster holding every runner: a common loading c adds c A to
    every performance and cancels. Uncentred it widened the lattice until
    the forward raised (mass 1.0247 at 40), the Jacobian moved and the
    inverse refused."""
    mu = np.array([-0.8, -0.3, 0.0, 0.2, 0.9])
    cl, D = [0] * 5, [1.0] * 5
    L = np.array([0.4, 0.2, 0.3, 0.5, 0.1])
    p0 = block_race_probabilities(mu, cl, L, D)
    J0 = block_race_jacobian(mu, cl, L, D)
    a0 = abilities_from_block_race(p0, cl, L, D)[0]
    tol = 1e-12 if c < 100 else 1e-9   # L + 1e6 itself rounds L by 1e-10
    assert np.abs(block_race_probabilities(mu, cl, L + c, D) - p0).max() < tol
    assert np.abs(block_race_jacobian(mu, cl, L + c, D) - J0).max() < tol
    assert np.abs(abilities_from_block_race(p0, cl, L + c, D)[0] - a0).max() < 1e-8


# --- #780: a scalar tree leaf loading is one loading for every runner ---

def _tree_780():
    mu = np.array([-.5, -.1, .2, .4])
    cluster = np.array([0, 0, 1, 1])
    D = np.array([.8, .9, 1., 1.1])
    parent = np.array([2, 2, -1])
    return mu, cluster, D, parent


@pytest.mark.parametrize("strength", [[0., 0., .4], [0., 0., 0.]])
@pytest.mark.parametrize("which", ["probabilities", "jacobian"])
def test_tree_scalar_loading_matches_expanded(which, strength):
    """Python indexed the 0-d loading by runner order and raised
    IndexError; the scalar must now price exactly as its expansion."""
    from winning.factor import blocks as B
    f = getattr(B, f"tree_race_{which}")
    mu, cluster, D, parent = _tree_780()
    lam = np.array(strength)
    expanded = f(mu, cluster, np.full(4, .3), D, parent, lam, points=129)
    scalar = f(mu, cluster, .3, D, parent, lam, points=129)
    assert np.array_equal(scalar, expanded)
    assert np.array_equal(f(mu, cluster, np.float64(.3), D, parent, lam,
                            points=129), expanded)


def test_tree_expanded_loading_reference_unchanged():
    """The vector spelling the fix leaves alone, at the issue's values."""
    from winning.factor.blocks import tree_race_probabilities
    mu, cluster, D, parent = _tree_780()
    p = tree_race_probabilities(mu, cluster, [.3] * 4, D, parent,
                                np.array([0., 0., .4]), points=129)
    assert np.allclose(p, [0.42684000, 0.25205727, 0.17948344, 0.14161929],
                       atol=1e-7)


def test_structured_tree_scalar_loading_matches_expanded():
    from winning.factor.races import race_probabilities
    from winning.factor.structures import Tree
    mu, cluster, D, parent = _tree_780()
    lam = np.array([0., 0., .4])
    a = race_probabilities(mu, structure=Tree(cluster, .3, D, parent, lam),
                           points=129)
    b = race_probabilities(mu, structure=Tree(cluster, np.full(4, .3), D,
                                              parent, lam), points=129)
    assert np.array_equal(a, b)


def test_tree_rank_r_loading_still_refused():
    from winning.factor.blocks import tree_race_probabilities
    mu, cluster, D, parent = _tree_780()
    with pytest.raises(NotImplementedError, match="rank-one"):
        tree_race_probabilities(mu, cluster, np.full((4, 2), .3), D, parent,
                                np.array([0., 0., .4]), points=129)


# --- #787: rank-two block loadings are canonical per cluster -------------
_MU787 = [2.43, .27, -.01, -.19, 1.07, -1.03]
_CL787 = [0, 0, 0, 1, 1, 1]
_D787 = [1, 1.39, 1.48, .85, .94, .61]
_V787 = np.array([[.69, -.65], [.29, -.95], [-.69, -1.15],
                  [-.04, -1.28], [.61, -.27], [-.20, 2.02]])


def _rot45(V):
    c = np.sqrt(0.5)
    return V @ np.array([[c, -c], [c, c]])


def test_rank_two_rotation_is_the_same_race_787():
    from winning.factor import race_probabilities
    from winning.factor.structures import Blocks, Nested
    Vr = _rot45(_V787)
    assert np.allclose(Vr @ Vr.T, _V787 @ _V787.T, atol=1e-15)
    p = block_race_probabilities(_MU787, _CL787, _V787, _D787)
    q = block_race_probabilities(_MU787, _CL787, Vr, _D787)
    assert np.abs(p - q).max() < 1e-14            # was 5.2e-3
    s = race_probabilities(_MU787, structure=Blocks(_CL787, Vr, _D787))
    assert np.abs(s - p).max() < 1e-14
    g = 0.3 * np.ones(6)
    pn = nested_race_probabilities(_MU787, _CL787, _V787, _D787, coupling=g)
    qn = nested_race_probabilities(_MU787, _CL787, Vr, _D787, coupling=g)
    assert np.abs(pn - qn).max() < 1e-14
    sn = race_probabilities(_MU787, structure=Nested(_CL787, Vr, _D787, g))
    assert np.abs(sn - pn).max() < 1e-14


def test_rank_two_per_cluster_rotations_signs_swaps_787():
    rng = np.random.default_rng(787)
    for _ in range(20):
        n = int(rng.integers(3, 9))
        cl = rng.integers(0, 3, n)
        mu = rng.normal(size=n); D = rng.uniform(.4, 1.6, n)
        V = 0.7 * rng.normal(size=(n, 2))
        p = block_race_probabilities(mu, cl, V, D)
        Vr = V.copy()
        for c in np.unique(cl):
            Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
            Vr[cl == c] = V[cl == c] @ Q
        for W in (Vr, V * [-1, 1], V[:, ::-1]):
            q = block_race_probabilities(mu, cl, W, D)
            assert np.abs(p - q).max() < 1e-13


def test_rank_two_converges_to_the_canonical_answer_787():
    # the product grid at 45 degrees to the principal axes: the default
    # error on the issue fixture fell from 5.7e-3 (caller's axes) to
    # 5.7e-4, and refining qa converges from either basis
    ref = block_race_probabilities(_MU787, _CL787, _V787, _D787, qa=61)
    for V in (_V787, _rot45(_V787)):
        p = block_race_probabilities(_MU787, _CL787, V, _D787)
        assert np.abs(p - ref).max() < 1e-3
        p = block_race_probabilities(_MU787, _CL787, V, _D787, qa=31)
        assert np.abs(p - ref).max() < 1e-4


def test_rank_deficient_rank_two_is_rank_one_787():
    v = _V787[:, 0]
    p1 = block_race_probabilities(_MU787, _CL787, np.sqrt(5) * v, _D787)
    p2 = block_race_probabilities(_MU787, _CL787, np.c_[v, 2 * v], _D787)
    assert np.abs(p1 - p2).max() < 1e-14
    # one cluster rank two, one cluster with no loading at all
    V = _V787.copy(); V[3:] = 0.0
    q = block_race_probabilities(_MU787, _CL787, _rot45(V), _D787)
    assert np.abs(q - block_race_probabilities(_MU787, _CL787, V,
                                               _D787)).max() < 1e-14


def test_rank_one_and_three_untouched_by_787():
    # pinned on main before #787: rank one, a zero-padded rank one and
    # rank three keep their exact rules
    r1 = [0.00244822624473019, 0.1180086074271331, 0.19963224309434435,
          0.14866389613894831, 0.03235943719199677, 0.4988875899028474]
    r3 = [0.00211546966829846, 0.08845359071227708, 0.15724230043463078,
          0.26820481052305434, 0.02406773393700397, 0.4599160947247354]
    v1 = _V787[:, 0]
    V3 = np.c_[_V787, [.3, -.2, .1, .4, -.1, .2]]
    assert np.allclose(block_race_probabilities(_MU787, _CL787, v1, _D787),
                       r1, rtol=0, atol=1e-15)
    assert np.allclose(block_race_probabilities(_MU787, _CL787, -v1, _D787),
                       r1, rtol=0, atol=1e-14)
    assert np.allclose(block_race_probabilities(_MU787, _CL787, V3, _D787),
                       r3, rtol=0, atol=1e-14)
    padded = np.c_[v1, np.zeros(6)]
    assert np.allclose(block_race_probabilities(_MU787, _CL787, padded,
                                                _D787), r1, rtol=0, atol=1e-14)


# --- #777: block/tree endpoints check their own lattice mass ---------------

def _structured_calls(mu, sd, points):
    """The three structured endpoints on one two-cluster field with zero
    loadings (a one-node rule), so all three price the same race. These
    direct-extension tests live here, not in test_fastrace_boundary.py,
    because the native CI job runs this file against the checkout's build."""
    fastrace = pytest.importorskip("fastrace")
    if not hasattr(fastrace, "tree_race"):
        pytest.skip("fastrace without the structured kernels")
    n = len(mu)
    v = np.zeros(n)
    starts = np.array([0, 1], dtype=np.int64)
    an, aw = np.array([0.0]), np.array([1.0])
    parent = np.array([2, 2, -1], dtype=np.int64)
    lam = np.array([0.0, 0.0, 0.5])
    return {
        "block_race": lambda **kw: fastrace.block_race(
            mu, sd, v, starts, an, aw, points, np.nan, np.nan, **kw),
        "block_race_r": lambda **kw: fastrace.block_race_r(
            mu, sd, v[:, None], starts, an[:, None], aw, points,
            np.nan, np.nan, **kw),
        "tree_race": lambda **kw: fastrace.tree_race(
            mu, sd, v, starts, parent, lam, an, aw, points, -8.0, 8.0,
            **kw),
    }


@pytest.mark.parametrize("name", ["block_race", "block_race_r", "tree_race"])
@pytest.mark.parametrize("mu, sd, points", [
    ([0.0, -0.2], [0.01, 1.0], 257),    # the issue's default-size field
    ([0.0, 0.0], [1.0, 1.0], 3),        # raw mass 3.19
    ([0.0, 0.0], [1.0, 1.0], 2),        # raw mass 1.6e-13
])
def test_structured_endpoints_refuse_a_mass_defect_777(name, mu, sd, points):
    """These returned raw lattice integrals summing to 1.07, 3.19 and
    1.6e-13 as if they were probabilities; the Python front door refuses
    the same defect, and so does the endpoint now."""
    call = _structured_calls(np.array(mu), np.array(sd), points)[name]
    with pytest.raises(RuntimeError, match="lattice captured total mass"):
        call()
    raw = np.asarray(call(mass_tol=np.inf))     # explicit opt-out: raw
    assert abs(raw.sum() - 1.0) > 5e-3


@pytest.mark.parametrize("name", ["block_race", "block_race_r", "tree_race"])
def test_structured_endpoints_resolved_race_unchanged_777(name):
    """A well-resolved lattice passes the check and is returned raw,
    bit for bit as with the check switched off."""
    call = _structured_calls(np.array([0.0, -0.2]), np.ones(2), 257)[name]
    p = np.asarray(call())
    assert np.array_equal(p, np.asarray(call(mass_tol=np.inf)))
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.allclose(p, [0.55623146, 0.44376854], atol=1e-8)
    for tol in (np.nan, -1e-3):
        with pytest.raises(ValueError, match="mass_tol"):
            call(mass_tol=tol)


def test_front_door_keeps_its_own_mass_tolerance_777():
    """winning checks the combined mass itself: a trial state inside an
    inverse runs under the looser tolerance, which the native check at
    the strict default must not pre-empt. points=6 captures mass 1.009.
    Runs on both paths; it is the compiled one that #777 could break."""
    from winning.factor import blocks as B
    args = (np.array([0.0, 0.2, 0.1]), np.array([0, 1, 1]),
            np.full(3, 0.3), np.ones(3))
    with pytest.raises(RuntimeError, match="block race lattice captured "
                       r"total mass 1\.0090 \(defect 9\.0\de-03"):
        B.block_race_probabilities(*args, points=6)
    with B._trial_mass_tolerance():
        p = B.block_race_probabilities(*args, points=6)
    assert abs(p.sum() - 1.0) < 1e-12
