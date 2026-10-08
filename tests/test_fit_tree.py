"""fit_tree: dense covariance onto a KNOWN genealogy (#622).

The genealogy is a parent array; every entrant is a Tree leaf and every
clade (an entrant with its descendants) an internal node. References:
exact recovery of an in-grammar C, a dense Lawson-Hanson NNLS over the
explicit pair design (independent of the O(n^2) normal-equation
assembly), and fixed-seed Monte Carlo of C for the priced race.
"""
import numpy as np
import pytest
from scipy.optimize import nnls

from winning import race_probabilities
from winning.factor import fit_tree
from winning.factor.structures import Tree


def _ancestry(p):
    n = len(p)
    anc = np.zeros((n, n), bool)
    for i in range(n):                       # parents precede children below
        if p[i] >= 0:
            anc[i] = anc[p[i]]
        anc[i, i] = True
    return anc


def _genealogy(n, rng, roots=1, window=None):
    p = np.array([-1 if i < roots else
                  rng.integers(0 if window is None else max(0, i - window), i)
                  for i in range(n)])
    return p


def _in_grammar(p, rng, common=0.7):
    """C from the tree model with only IDENTIFIABLE clades nonzero."""
    n = len(p)
    anc = _ancestry(p)
    size = anc.sum(0)
    ident = (size >= 2) & (size <= n - 2)
    roots = np.flatnonzero(p < 0)
    if len(roots) == 2 and ident[roots].all():
        ident[roots[1]] = False
    s = rng.uniform(0.2, 1.0, n) * ident
    D = rng.uniform(0.3, 1.5, n)
    C = (anc * s) @ anc.T + np.diag(D) + common
    return C, s, D


@pytest.mark.parametrize("n,roots,seed", [(3, 1, 0), (6, 1, 1), (25, 1, 2),
                                          (25, 3, 3), (12, 2, 4), (60, 1, 5)])
def test_exact_recovery_from_a_known_tree(n, roots, seed):
    rng = np.random.default_rng(seed)
    p = _genealogy(n, rng, roots=roots)
    C, s, D = _in_grammar(p, rng)
    V, Dh, tree, rep = fit_tree(C, p)
    assert V.shape == (n, 0)
    assert isinstance(tree, Tree)
    np.testing.assert_allclose(Dh, D, atol=1e-10)
    np.testing.assert_allclose(tree.D, D, atol=1e-10)
    np.testing.assert_allclose(tree.strength[n:2 * n] ** 2, s, atol=1e-10)
    assert rep["projected_residual_rel"] < 1e-12
    assert rep["contrast_residual_max"] < 1e-10
    assert rep["clades_zero"] == 0 and rep["floored"] == 0


def test_tree_mapping_is_the_documented_one():
    # entrant j -> leaf j; clade(j) -> node n + j; roots under node 2n
    p = np.array([-1, 0, 0, 1, 1, -1])
    n = len(p)
    rng = np.random.default_rng(0)
    C, _, _ = _in_grammar(p, rng)
    _, _, tree, _ = fit_tree(C, p)
    assert list(tree.parent[:n]) == [n + j for j in range(n)]
    assert list(tree.parent[n:2 * n]) == [2 * n, n + 0, n + 0, n + 1, n + 1,
                                          2 * n]
    assert tree.parent[2 * n] == -1 and tree.strength[2 * n] == 0.0
    assert np.all(tree.strength[:n] == 0.0)


def test_internal_entrants_are_priced_as_leaves_of_their_own_clade():
    # entrant 0 is the ancestor of all others; the clade {1, 2} shares an
    # edit. Var(theta1 - theta2) excludes it, Var(theta0 - theta1) has it.
    p = np.array([-1, 0, 1, 0])     # 0 -> 1 -> 2, 0 -> 3
    s1, D = 0.8, np.array([0.5, 0.6, 0.7, 0.4])
    anc = _ancestry(p)
    s = np.array([0.0, s1, 0.0, 0.0])
    C = (anc * s) @ anc.T + np.diag(D)
    _, Dh, tree, rep = fit_tree(C, p)
    np.testing.assert_allclose(Dh, D, atol=1e-12)
    assert tree.strength[4 + 1] ** 2 == pytest.approx(s1, abs=1e-12)


def test_matches_dense_nnls_on_an_out_of_grammar_covariance():
    # an arbitrary PSD C: some clades are driven to zero. Independent
    # reference: Lawson-Hanson on the explicit n(n-1)/2-row pair design.
    rng = np.random.default_rng(7)
    n = 14
    p = _genealogy(n, rng)
    X = rng.normal(size=(n, 3))
    C = X @ X.T + np.diag(rng.uniform(0.2, 1.0, n))
    _, Dh, tree, rep = fit_tree(C, p)
    anc = _ancestry(p)
    size = anc.sum(0)
    J = np.flatnonzero((size >= 2) & (size <= n - 2))
    rows, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.zeros(n + len(J))
            r[i] = r[j] = 1.0
            r[n:] = anc[i, J] != anc[j, J]
            rows.append(r)
            b.append(C[i, i] + C[j, j] - 2 * C[i, j])
    A, b = np.array(rows), np.array(b)
    c = np.diag(C)
    floor = 1e-6 * np.maximum(c, 1e-6 * c.mean())
    lb = np.r_[floor, np.zeros(len(J))]
    y, _ = nnls(A, b - A @ lb)
    x = lb + y
    np.testing.assert_allclose(Dh, x[:n], atol=1e-9)
    np.testing.assert_allclose(tree.strength[n + J] ** 2, x[n:], atol=1e-9)
    assert rep["clades_zero"] == int(np.sum(x[n:] <= 0))
    assert rep["projected_residual_rel"] > 1e-3          # really out of grammar


def test_permutation_invariance():
    rng = np.random.default_rng(11)
    n = 30
    p = _genealogy(n, rng, roots=2)
    X = rng.normal(size=(n, 2)) * 0.4
    C = _in_grammar(p, rng)[0] + X @ X.T             # out of grammar too
    mu = rng.normal(size=n) * 0.5
    _, D, tree, rep = fit_tree(C, p)
    perm = rng.permutation(n)
    inv = np.argsort(perm)                            # old index -> new index
    pp = np.where(p[perm] >= 0, inv[np.maximum(p[perm], 0)], -1)
    _, Dp, treep, repp = fit_tree(C[np.ix_(perm, perm)], pp)
    np.testing.assert_allclose(Dp, D[perm], atol=1e-9)
    np.testing.assert_allclose(treep.strength[n:2 * n],
                               tree.strength[n:2 * n][perm], atol=1e-9)
    assert repp["projected_residual_rel"] == pytest.approx(
        rep["projected_residual_rel"], abs=1e-12)
    pr = race_probabilities(mu, structure=tree)
    prp = race_probabilities(mu[perm], structure=treep)
    np.testing.assert_allclose(prp, pr[perm], atol=1e-10)


def test_priced_race_agrees_with_monte_carlo_of_C():
    rng = np.random.default_rng(3)
    n = 10
    p = _genealogy(n, rng, roots=2)
    C, _, _ = _in_grammar(p, rng)
    mu = rng.normal(size=n) * 0.6
    _, _, tree, _ = fit_tree(C, p)
    pr = race_probabilities(mu, structure=tree)
    draws = 400_000
    Y = mu + np.random.default_rng(123).standard_normal((draws, n)) \
        @ np.linalg.cholesky(C).T
    mc = np.bincount(Y.argmin(1), minlength=n) / draws
    # 4.5 standard errors per runner at 4e5 draws
    se = np.sqrt(mc * (1 - mc) / draws)
    assert np.all(np.abs(pr - mc) < 4.5 * se + 1e-4), np.abs(pr - mc) / se


def test_common_shift_is_choice_irrelevant():
    rng = np.random.default_rng(5)
    p = _genealogy(15, rng)
    C, _, _ = _in_grammar(p, rng, common=0.0)
    _, D0, t0, _ = fit_tree(C, p)
    _, D1, t1, _ = fit_tree(C + 3.0, p)
    np.testing.assert_allclose(D1, D0, atol=1e-10)
    np.testing.assert_allclose(t1.strength, t0.strength, atol=1e-10)


@pytest.mark.parametrize("parent,msg", [
    ([-1, 0], "one entry per entrant"),
    ([[-1, 0, 1]], "one entry per entrant"),
    ([-1, 0, 3], "not -1 or an entrant index"),
    ([-1, 0, -2], "not -1 or an entrant index"),
    ([-1, 2, 1], "cycle"),
    ([0, 1, 2], "cycle"),                          # self-parent
    ([-1, 0.5, 0], "integers"),
    ([-1, np.nan, 0], "integers"),
])
def test_invalid_parent_arrays_refused(parent, msg):
    C = np.eye(3) + 0.2
    with pytest.raises(ValueError, match=msg):
        fit_tree(C, np.asarray(parent))


def test_integral_float_parent_accepted():
    rng = np.random.default_rng(0)
    p = np.array([-1, 0, 0, 1, 1, 2])
    C, s, D = _in_grammar(p, rng)
    _, Dh, _, _ = fit_tree(C, p.astype(float))
    np.testing.assert_allclose(Dh, D, atol=1e-10)


def test_invalid_covariance_and_k_refused():
    p = np.array([-1, 0, 0])
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_tree(np.array([[1.0, 2, 0], [2, 1, 0], [0, 0, 1]]), p)
    with pytest.raises(ValueError, match="square"):
        fit_tree(np.ones((3, 2)), p)
    with pytest.raises(NotImplementedError, match="factors PLUS a tree"):
        fit_tree(np.eye(3), p, k=2)
    with pytest.raises(ValueError, match="integer k"):
        fit_tree(np.eye(3), p, k=-1)


def test_size_cap_refuses_instead_of_blowing_up():
    n = 30
    p = np.r_[-1, np.zeros(n - 1, int)]
    with pytest.raises(ValueError, match="max_n"):
        fit_tree(np.eye(n), p, max_n=20)


def test_tiny_fields():
    _, D, tree, rep = fit_tree(np.array([[2.0]]), [-1])
    assert D[0] == pytest.approx(2.0)
    np.testing.assert_allclose(race_probabilities([0.0], structure=tree), [1])
    C = np.array([[1.0, 0.3], [0.3, 2.0]])
    _, D, tree, rep = fit_tree(C, [-1, 0])
    assert D.sum() == pytest.approx(3.0 - 0.6)
    assert rep["projected_residual_rel"] < 1e-12
