"""Block-grammar boundaries the browser's structureCov now shares (#792, #822).

#792: a cluster vector shorter than mu priced only the labelled runners
([0.5, 0.5, 0, 0] for four exchangeable runners) or raised a misleading
lattice-mass error; an overlong one indexed past mu.
#822: structure_variances walked a forest as if a zero-strength common
root joined it (and a cyclic parent spun its walk forever).
"""
import numpy as np
import pytest

from winning import race_probabilities
from winning.factor import blocks as B
from winning.factor.structures import Blocks, Nested, Tree, structure_variances

MU4 = np.array([0.0, 0.1, 0.2, 0.3])
ONES = np.ones(4)
PARENT, STRENGTH = [2, 2, -1], [0.0, 0.0, 1.0]

CALLS = {
    "block": lambda c: B.block_race_probabilities(MU4, c, np.zeros(4), ONES),
    "nested": lambda c: B.nested_race_probabilities(
        MU4, c, np.zeros(4), ONES, coupling=0.3 * ONES),
    "nested-shortcut": lambda c: B.nested_race_probabilities(
        MU4, c, np.zeros(4), ONES, coupling=None),
    "block-jacobian": lambda c: B.block_race_jacobian(MU4, c, np.zeros(4), ONES),
    "nested-jacobian": lambda c: B.nested_race_jacobian(
        MU4, c, np.zeros(4), ONES, coupling=0.3 * ONES),
    "tree": lambda c: B.tree_race_probabilities(
        MU4, c, np.zeros(4), ONES, PARENT, STRENGTH),
    "tree-jacobian": lambda c: B.tree_race_jacobian(
        MU4, c, np.zeros(4), ONES, PARENT, STRENGTH),
    "Blocks": lambda c: race_probabilities(
        MU4, structure=Blocks(c, np.zeros(4), ONES)),
    "Nested": lambda c: race_probabilities(
        MU4, structure=Nested(c, np.zeros(4), ONES, coupling=None, gamma=0.0)),
}


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("cluster", [[0, 1], [0, 0, 1, 1, 1]])
def test_cluster_must_have_one_label_per_contestant_792(name, cluster):
    with pytest.raises(ValueError, match="one label per contestant"):
        CALLS[name](cluster)


def test_the_short_cluster_no_longer_drops_runners_792():
    # the valid four-label spelling of the same exchangeable race
    p = B.block_race_probabilities(np.zeros(4), [0, 0, 1, 1], np.zeros(4), ONES)
    assert np.allclose(p, 0.25, atol=1e-12)


def test_valid_cluster_spellings_are_unchanged_792():
    # labels of any comparable kind, one per contestant, still price
    mu = np.array([-0.3, 0.0, 0.2, 0.5])
    a = B.block_race_probabilities(mu, [0, 0, 1, 1], [0.4, 0.2, 0.3, 0.5], ONES)
    b = B.block_race_probabilities(mu, ["x", "x", "y", "y"],
                                   [0.4, 0.2, 0.3, 0.5], ONES)
    c = B.block_race_probabilities(mu, np.array([7, 7, 9, 9]),
                                   [0.4, 0.2, 0.3, 0.5], ONES)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    t = B.tree_race_probabilities(mu, [0, 0, 1, 1], [0.4, 0.2, 0.3, 0.5],
                                  ONES, PARENT, STRENGTH)
    assert abs(t.sum() - 1) < 1e-12


def test_structure_variances_refuses_a_forest_822():
    forest = Tree([0, 1, 2], [0, 0, 0], [1, 1, 1], [3, 3, 4, -1, -1],
                  [0, 0, 0, 2, 0])
    with pytest.raises(ValueError, match="exactly one root"):
        structure_variances(forest)
    with pytest.raises(ValueError, match="exactly one root"):
        race_probabilities(np.zeros(3), structure=forest)


def test_structure_variances_refuses_a_cycle_822():
    cyc = Tree([0, 1, 2], [0, 0, 0], [1, 1, 1], [3, 3, 4, 4, 3, -1],
               [0, 0, 0, 1, 1, 0])
    with pytest.raises(ValueError, match="cycle"):
        structure_variances(cyc)


def test_structure_variances_of_a_valid_tree_is_unchanged_822():
    # the forest above joined by an explicit zero-strength root 4:
    # leaves 0 and 1 carry node 3's 2^2, leaf 2 its idiosyncratic 1
    t = Tree([0, 1, 2], [0, 0, 0], [1, 1, 1], [3, 3, 4, 4, -1],
             [0, 0, 0, 2, 0])
    assert structure_variances(t).tolist() == [5.0, 5.0, 1.0]
    deep = Tree([0, 0, 1], [0.5, 0.5, 0.5], [1, 1, 1], [2, 2, -1], [0, 0, 3])
    assert structure_variances(deep).tolist() == [1.25, 1.25, 1.25]
