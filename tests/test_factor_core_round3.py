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
