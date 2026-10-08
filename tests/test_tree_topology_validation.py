"""A tree race's parent vector is a tree: cycles used to spin forever in
`while parent[u] >= 0` (#328). Each malformed topology is refused at once."""
import numpy as np
import pytest

from winning.factor.blocks import tree_race_jacobian, tree_race_probabilities


@pytest.mark.parametrize("parent,strength,match", [
    ([1, 0], [0.2, 0.2], "root"),                 # 0 -> 1 -> 0, no root
    ([2, 2, 2], [0, 0, 0.3], "cycle"),            # self-parent
    ([2, 2, -1, 3], [0, 0, 0.3, 0.1], "cycle"),   # detached self-loop
    ([2, 2, 5], [0, 0, 0.3], "node index"),       # out of range
    ([2, 2, -1], [0, 0], "strength"),             # length mismatch
    ([2, -1, -1], [0, 0, 0.3], "one root"),       # two roots
])
def test_malformed_topologies_are_refused(parent, strength, match):
    args = ([0.0, 0.0], [0, 1], [0.0, 0.0], [1.0, 1.0], parent, strength)
    for fn in (tree_race_probabilities, tree_race_jacobian):
        with pytest.raises(ValueError, match=match):
            fn(*args, points=33, qa=3)


def test_a_valid_tree_still_prices():
    p = tree_race_probabilities([0.0, 0.3], [0, 1], [0.2, 0.2], [1.0, 1.0],
                                [2, 2, -1], [0, 0, 0.5], points=129, qa=7)
    assert abs(p.sum() - 1) < 1e-12 and p[0] > p[1]
