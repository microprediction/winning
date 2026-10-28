"""Regression tests for the factor-core round-3 issue batch."""
import warnings

import numpy as np
import pytest

from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes, win_probabilities_factor)


# #374: the pair closed form is exact only for a Gaussian factor law; a
# centred two-point rule has the same two moments and a different map.
@pytest.mark.parametrize("F, D, truth", [
    ([[-1.0], [1.0]], 0.05, 0.4),     # centred, mean zero, variance one
    ([[0.0], [2.0]], 0.2, 0.3),       # translated
])
def test_pair_inverse_is_certified_by_the_forward(F, D, truth):
    V = np.array([[0.0], [1.0]]); D = np.full(2, D)
    F = np.array(F); W = np.array([0.5, 0.5])
    t = win_probabilities_factor(np.array([-truth, truth]), V, D, F, W,
                                 points=1025)
    mu, info = abilities_from_probabilities_factor(t, V, D, F, W,
                                                   points=1025,
                                                   return_info=True)
    r = win_probabilities_factor(mu, V, D, F, W, points=1025)
    actual = float(np.abs(np.log(r / t)).max())
    assert info["converged"] and actual < 1e-6
    assert abs(info["residual"] - actual) < 1e-9
    assert np.allclose(mu, [-truth, truth], atol=1e-6)


def test_pair_inverse_on_a_hermite_rule_round_trips():
    V = np.array([[0.0], [1.0]]); D = np.full(2, 0.05)
    F, W = hermite_nodes(1)
    t = win_probabilities_factor(np.array([-0.3, 0.3]), V, D, F, W)
    mu, info = abilities_from_probabilities_factor(t, V, D, F, W,
                                                   return_info=True)
    assert info["converged"]
    assert np.allclose(mu, [-0.3, 0.3], atol=1e-5)


# #445: delta is the bulk window's tail mass in (0, 0.5); -1 hung forever.
@pytest.mark.parametrize("delta", [-1, -1e-9, 0, 0.5, 1, 2, np.nan, np.inf,
                                   "1e-2", None, True])
def test_bulk_window_delta_is_validated(delta):
    from winning.factor.races import race_probabilities
    with pytest.raises(ValueError, match="delta must be"):
        race_probabilities([-1.0, 0.0, 1.0], D=[1.0, 1.0, 1.0], delta=delta,
                           points=257)


def test_bulk_window_delta_accepts_ordinary_values():
    from winning.factor.races import race_probabilities
    a = race_probabilities([-1.0, 0.0, 1.0], D=[1.0, 1.0, 1.0], delta=1e-3)
    b = race_probabilities([-1.0, 0.0, 1.0], D=[1.0, 1.0, 1.0],
                           delta=np.float32(1e-6))
    assert np.all(np.isfinite(a)) and np.abs(a - b).max() < 1e-6


# #416: a node of weight 0 or 1e-300 far out must not change the price.
@pytest.mark.parametrize("w", [0.0, 1e-300, 1e-17])
def test_negligible_factor_nodes_are_no_ops(w):
    from winning.factor import race_probabilities
    mu = np.array([-0.8, -0.1, 0.5, 1.0])
    V = np.array([[-1.0], [-0.2], [0.4], [0.8]]); D = np.ones(4)
    F0 = np.array([[-1.0], [1.0]]); W0 = np.array([0.5, 0.5])
    F = np.array([[-1.0], [1.0], [1000.0]]); W = np.array([0.5, 0.5, w])
    base = win_probabilities_factor(mu, V, D, F0, W0, points=257)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = win_probabilities_factor(mu, V, D, F, W, points=257)
        b = race_probabilities(mu, V=V, D=D, F=F, W=W, points=257)
    assert np.abs(a - base).max() < 1e-12
    assert np.abs(b - race_probabilities(mu, V=V, D=D, F=F0, W=W0,
                                         points=257)).max() < 1e-12


def test_factor_law_keeps_ordinary_rules_intact():
    from winning.shapes import as_factor_law
    for Q in (7, 15):
        F, W = hermite_nodes(1, Q)
        F2, W2 = as_factor_law(F, W)
        assert len(W2) == Q and np.allclose(W2, W / W.sum(), rtol=1e-14, atol=0)


# #384: subnormal longshots keep their Harville memberships.
@pytest.mark.parametrize("t", [1e-301, 1e-310, 0.0])
@pytest.mark.parametrize("k", [2, 3])
def test_harville_topk_sums_to_k_with_subnormal_shares(t, k):
    from winning.factor.races import plackett_luce_topk_probabilities
    q = plackett_luce_topk_probabilities(np.array([1.0, t, t, t]), k)
    assert abs(q.sum() - k) < 1e-12
    assert np.allclose(q[1:], (k - 1) / 3, atol=1e-12)


# #438: the pure classic path refuses non-finite offsets like the kernel.
@pytest.mark.parametrize("rust", [True, False])
@pytest.mark.parametrize("x", [np.inf, -np.inf, np.nan])
def test_classic_offsets_must_be_finite_on_every_backend(rust, x):
    from winning.classic.lattice import (skew_normal_density,
                                         state_prices_from_offsets)
    d = skew_normal_density(50, 0.1)
    from winning import rustconfig
    prev = rustconfig.rust_active()
    rustconfig.use_rust(rust)
    try:
        with pytest.raises(ValueError, match=r"offsets\[1\] = .* is not finite"):
            state_prices_from_offsets(d, [0.0, x])
        assert abs(sum(state_prices_from_offsets(d, [0, 1, 2])) - 1) < 1e-12
    finally:
        rustconfig.use_rust(prev)


# #328 / #459: linkage topology and heights are checked at the constructor,
# and fractional explicit parents are refused rather than truncated.
@pytest.mark.parametrize("Z, msg", [
    ([[0, 1, -0.5, 2], [2, 3, 0.8, 3]], "height"),
    ([[0, 1, np.nan, 2], [2, 3, 0.8, 3]], "height"),
    ([[3, 0, 0.5, 2], [1, 2, 0.8, 3]], "before it is formed"),
    ([[0, 0, 0.5, 2], [1, 2, 0.8, 3]], "with itself"),
    ([[0, 1, 0.5, 2], [0, 2, 0.8, 3]], "reuses"),
    ([[0, 1.5, 0.5, 2], [2, 3, 0.8, 3]], "whole numbers"),
    ([[0, 1, 0.5, 2], [2, 9, 0.8, 3]], "before it is formed"),
])
def test_linkage_is_validated_at_the_constructor(Z, msg):
    from winning.factor.structures import Tree
    with pytest.raises(ValueError, match=msg):
        Tree.from_linkage(np.array(Z, float))


def test_valid_linkage_still_builds():
    from winning.factor.structures import Tree
    t = Tree.from_linkage(np.array([[0, 1, 0.5, 2], [2, 3, 0.8, 3]]))
    assert t.parent.tolist() == [3, 3, 4, 4, -1]
    assert Tree.from_linkage(np.zeros((0, 4))).parent.tolist() == [-1]


def test_fractional_tree_parent_is_refused():
    from winning.factor.blocks import tree_race_probabilities
    from winning.factor.structures import Tree
    args = dict(mu=[0, 0.3, 0.6], cluster=[0, 0, 1], loading=[1, 1, 1],
                D=[1, 1, 1], strength=[.3, .3, .3])
    with pytest.raises(ValueError, match="whole node indices"):
        tree_race_probabilities(parent=[2.7, 2, -1], **args)
    with pytest.raises(ValueError, match="whole node indices"):
        Tree(cluster=[0, 1, 0], loading=[0, 0, 0], D=[1, 1, 1],
             parent=[2.7, 2, -1], strength=[.2, .2, .2])
    p = tree_race_probabilities(parent=[2.0, 2.0, -1.0], **args)
    assert np.allclose(p, tree_race_probabilities(parent=[2, 2, -1], **args))


# #372: an explicit label -> leaf map makes Tree pricing spelling-free.
def test_tree_leaves_map_makes_relabeling_invariant():
    from winning import Tree
    from winning.factor.blocks import (tree_race_jacobian,
                                       tree_race_probabilities)
    from winning.factor.races import abilities_from_race, race_probabilities
    mu = [-0.5, -0.1, 0.2, 0.4]; D = [0.8, 0.9, 1.0, 1.1]
    L = [0.2, 0.3, -0.1, 0.4]; pa = [3, 3, 4, 4, -1]
    st = [0, 0, 0, 0.6, 0.4]
    a = tree_race_probabilities(mu, [0, 0, 1, 2], L, D, pa, st)
    b = tree_race_probabilities(mu, [30, 30, 10, 20], L, D, pa, st,
                                leaves=[30, 10, 20])
    assert np.abs(a - b).max() == 0.0
    # the default is unchanged: sorted labels are the leaf nodes
    c = tree_race_probabilities(mu, [30, 30, 10, 20], L, D, pa, st)
    assert np.abs(a - c).max() > 1e-2
    Ja = tree_race_jacobian(mu, [0, 0, 1, 2], L, D, pa, st)
    Jb = tree_race_jacobian(mu, ["b", "b", "c", "a"], L, D, pa, st,
                            leaves=["b", "c", "a"])
    assert np.abs(Ja - Jb).max() == 0.0
    T = Tree([30, 30, 10, 20], L, D, pa, st, leaves=[30, 10, 20])
    T0 = Tree([0, 0, 1, 2], L, D, pa, st)
    p = race_probabilities(mu, structure=T)
    assert np.abs(p - race_probabilities(mu, structure=T0)).max() == 0.0
    m = abilities_from_race(p, structure=T)
    assert np.abs(m - (np.array(mu) - np.mean(mu))).max() < 1e-6


@pytest.mark.parametrize("leaves, msg", [
    ([30, 10], "does not map"), ([30, 10, 20, 40], "carries no runner"),
    ([30, 30, 20], "twice")])
def test_tree_leaves_map_is_validated(leaves, msg):
    from winning import Tree
    with pytest.raises(ValueError, match=msg):
        Tree([30, 30, 10, 20], [0.2] * 4, [1.0] * 4, [3, 3, 4, 4, -1],
             [0, 0, 0, .6, .4], leaves=leaves)


# #592: the floor holds in the final normalised target, whatever the units.
@pytest.mark.parametrize("c", [1e-3, 1.0, 1e3])
def test_topk_target_floor_holds_after_renormalisation(c):
    from winning.factor.topk import abilities_from_topk, top_k_probabilities
    mu, info = abilities_from_topk(np.array([.7, .3, 0.]) * c, 1,
                                   D=np.ones(3), target_floor=.2,
                                   return_info=True)
    q = top_k_probabilities(mu, 1, D=np.ones(3))
    assert info["converged"] and info["floored"].tolist() == [0, 0, 1]
    assert np.allclose(q, [0.56, 0.24, 0.2], atol=1e-7)


def test_race_target_floor_is_scale_free_and_respected():
    from winning.factor.races import abilities_from_race, race_probabilities
    out = [abilities_from_race(np.array([c, 0.]), D=[.5, .5],
                               target_floor=1e-6) for c in (1e-6, 1, 1e6)]
    assert max(np.abs(o - out[0]).max() for o in out) < 1e-9
    assert abs(race_probabilities(out[0], D=[.5, .5])[1] - 1e-6) < 1e-12
    with pytest.raises(ValueError, match="infeasible"):
        abilities_from_race([.5, .5, 0], target_floor=.4)


# #471: an exactly common loading column vanishes at any finite scale, and
# a column whose centring overflows is centred in units of its largest entry.
@pytest.mark.parametrize("c", [1e20, 1e100, 1e307, 1e308])
def test_common_loading_column_is_inert_at_any_scale(c):
    from winning.factor.races import abilities_from_race, race_probabilities
    from winning.factor.topk import (abilities_from_topk, top_k_jacobians,
                                     top_k_probabilities)
    n = 9
    mu = (np.arange(n) - 4) * 0.1; D = np.ones(n); V = np.full((n, 1), c)
    F = np.array([[-1.0], [1.0]]); W = np.array([0.5, 0.5])
    ind = race_probabilities(mu, D=D, points=257)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = race_probabilities(mu, V=V, D=D, F=F, W=W, points=257)
        q = top_k_probabilities(mu, 3, D=D, V=V)
        m = abilities_from_race(ind, V=V, D=D, F=F, W=W, points=257)
    assert np.abs(p - ind).max() < 1e-15
    assert np.abs(q - top_k_probabilities(mu, 3, D=D)).max() < 1e-12
    assert np.abs(m - (mu - mu.mean())).max() < 1e-6
    J = top_k_jacobians(mu, 3, D=D, V=V)
    J0 = top_k_jacobians(mu, 3, D=D)
    assert np.abs(np.asarray(J[0]) - np.asarray(J0[0])).max() < 1e-12
    t = top_k_probabilities(mu, 3, D=D)
    mt = abilities_from_topk(t, 3, D=D, V=V)
    assert np.abs(mt - (mu - mu.mean())).max() < 1e-6


def test_overflowing_loading_with_compensating_nodes_prices_the_same_race():
    from winning.factor.races import race_probabilities
    mu = np.array([0., .3, -.2]); D = np.ones(3); W = np.array([.5, .5])
    ref = race_probabilities(mu, V=np.array([1., 1., -1.]), D=D,
                             F=np.array([[-1.], [1.]]), W=W)
    c = 1e308
    got = race_probabilities(mu, V=c * np.array([1., 1., -1.]), D=D,
                             F=np.array([[-1.], [1.]]) / c, W=W)
    assert np.all(np.isfinite(got)) and np.abs(got - ref).max() < 1e-12


# #656: a target priced by the default hierarchical forward inverts with
# the same controls; the warm start's mass defect no longer aborts it.
_F656 = dict(mu=np.array([0.36, -0.92, 0.30, 0.26]), cluster=[0, 0, 1, 2],
             loading=[0.022, 1.28, -0.083, -0.04], D=[0.035, 25, 0.062, 0.015])


def test_block_inverse_round_trips_a_default_target():
    from winning.factor.blocks import (abilities_from_block_race,
                                       block_race_probabilities)
    f = _F656
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = block_race_probabilities(f["mu"], f["cluster"], f["loading"], f["D"])
        mu, res, _ = abilities_from_block_race(t, f["cluster"], f["loading"],
                                               f["D"], tol=1e-8)
    assert res < 1e-8 and np.abs(mu - f["mu"]).max() < 1e-6


@pytest.mark.parametrize("kind", ["blocks", "nested", "tree"])
def test_structured_inverse_round_trips_a_default_target(kind):
    from winning import Blocks, Nested, Tree
    from winning.factor.races import abilities_from_race, race_probabilities
    f = _F656
    s = {"blocks": Blocks(f["cluster"], f["loading"], f["D"]),
         "nested": Nested(f["cluster"], f["loading"], f["D"],
                          [0.05, -0.1, 0.08, 0.02], 0.5),
         "tree": Tree(f["cluster"], f["loading"], f["D"], [3, 3, 4, 4, -1],
                      [0, 0, 0, 0.2, 0])}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = race_probabilities(f["mu"], structure=s)
        mu, info = abilities_from_race(p, structure=s, return_info=True)
    assert info["converged"] and np.abs(mu - f["mu"]).max() < 1e-6


def test_block_mass_check_is_strict_outside_inverses():
    from winning.factor.blocks import _MASS_TOL
    assert _MASS_TOL.get() == 5e-3


# #666: positive semidefiniteness is judged in correlation units.
@pytest.mark.parametrize("S", [
    [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1e12]],
    [[0.0, 1e-309], [1e-309, 0.0]],
    [[0.0, 1.0], [1.0, 0.0]],
])
def test_covariance_validation_is_scale_free(S):
    from winning.factor.core import _validate_covariance, fit_covariance
    S = np.array(S)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        _validate_covariance(S)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        fit_covariance(S)


def test_covariance_validation_accepts_heterogeneous_scales():
    from winning.factor.core import _validate_covariance
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    for c in (1e-150, 1.0, 1e150):
        _validate_covariance(A * c)
    _validate_covariance(np.diag([1e-12, 1.0, 1e12]))
    _validate_covariance(np.zeros((2, 2)))


# #682: ordered_probabilities(k=1, cov=) is race_probabilities(cov=).
def test_ordered_k1_with_cov_routes_like_the_win_race():
    from winning.factor import race_probabilities
    from winning.factor.permutations import ordered_probabilities
    n = 8
    rng = np.random.default_rng(3)
    V0 = rng.normal(size=(n, 3)); d0 = 0.05 + rng.random(n)
    S = V0 @ V0.T + np.diag(d0); s = np.sqrt(np.diag(S))
    C = S / np.outer(s, s)
    mu = np.linspace(-0.6, 0.6, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(race_probabilities(mu, cov=C, points=257))
        q = ordered_probabilities(mu, k=1, cov=C, points=257)
        oracle = race_probabilities(mu, V=V0 / s[:, None], D=d0 / s**2,
                                    points=257)
    assert np.abs(p - q).max() == 0.0
    assert np.abs(q - oracle).max() < 1e-4


# #684: a scalar block loading means the same loading for every runner.
def test_scalar_block_loading_is_accepted_everywhere():
    from winning import Blocks
    from winning.factor.blocks import (abilities_from_block_race,
                                       block_race_probabilities)
    from winning.factor.races import abilities_from_race, race_probabilities
    mu = np.array([-.5, -.1, .2, .4]); cl = np.array([0, 0, 1, 1])
    D = np.array([.8, .9, 1., 1.1])
    p = block_race_probabilities(mu, cl, np.full(4, .3), D, points=129)
    assert np.abs(block_race_probabilities(mu, cl, .3, D, points=129)
                  - p).max() == 0.0
    m, res, _ = abilities_from_block_race(p, cl, .3, D, points=129)
    assert np.abs(m - (mu - mu.mean())).max() < 1e-6
    s = Blocks(cl, .3, D)
    assert np.abs(race_probabilities(mu, structure=s, points=129)
                  - p).max() < 1e-15
    assert np.abs(abilities_from_race(p, structure=s, points=129)
                  - (mu - mu.mean())).max() < 1e-6


# #465: fit_covariance(C + a 11') fits the same race for every a >= 0.
@pytest.mark.parametrize("a", [1e3, 1e6, 2.0 ** 30])
def test_fit_covariance_is_invariant_to_a_common_shift(a):
    from winning.factor.core import fit_covariance
    from winning.factor.races import race_probabilities
    rng = np.random.default_rng(1); n = 6
    A = rng.normal(size=(n, 2)); C0 = A @ A.T + np.diag(0.3 + rng.random(n))
    mu = np.linspace(-1, 1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = race_probabilities(mu, *fit_covariance(C0 + 1.0)[:2],
                                 *fit_covariance(C0 + 1.0)[2:])
        V, D, F, W = fit_covariance(C0 + a)
        got = race_probabilities(mu, V=V, D=D, F=F, W=W)
    assert np.abs(got - ref).max() < 1e-5


def test_fit_covariance_keeps_contrasts_under_a_huge_shift():
    from winning.factor.core import fit_covariance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V, D, _, _ = fit_covariance(np.eye(4) + 2.0 ** 51)
    S = V @ V.T + np.diag(D)
    d = np.diag(S)
    contrasts = d[:, None] + d[None, :] - 2 * S
    assert np.allclose(contrasts[~np.eye(4, dtype=bool)], 2.0, rtol=1e-5)


def test_fit_covariance_leaves_mixed_sign_fields_alone():
    from winning.factor.core import _without_common_shift
    C = np.array([[1.0, -0.2, 0.1], [-0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
    assert _without_common_shift(C) is C
    B = _without_common_shift(np.array([[1.0, 0.5], [0.5, 0.3]]))
    assert np.linalg.eigvalsh(B).min() > -1e-12


# #97: the bulk window covers each runner's own win integrand.
@pytest.mark.parametrize("gap", [8.0, 16.0, 20.0])
def test_bulk_window_keeps_rare_winners_relatively_accurate(gap):
    from scipy.integrate import quad
    from scipy.stats import norm
    from winning.factor.races import race_probabilities
    mu = np.array([0.0, gap, 2 * gap])

    def exact(i):
        f = lambda x: norm.pdf(x - mu[i]) * np.prod(
            [norm.sf(x - mu[j]) for j in range(3) if j != i])
        return quad(f, mu[i] - 60, mu[i] + 60, points=list(mu), limit=500,
                    epsabs=0, epsrel=1e-12)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = race_probabilities(mu, D=np.ones(3), points=257)
    for i in (1, 2):
        assert abs(p[i] / exact(i) - 1) < 1e-9


def test_bulk_window_edge_check_leaves_ordinary_fields_alone():
    from winning.factor import races
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 40))
        mu = rng.normal(size=n); s = np.exp(0.3 * rng.normal(size=n))
        x = races._bulk_window(mu[None, :], s, 257, 1e-12)
        assert len(x) == 257
