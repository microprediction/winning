"""Cluster labels are arbitrary comparable values, on BOTH paths.

The forward dispatch says so: every tree and block kernel in
`winning/factor/blocks.py` remaps them with
`np.unique(..., return_inverse=True)`. `structure_variances`, which the
generic inverse uses, cast them to `int` and used them directly as node
IDs, so a tree labelled 10/20 priced fine and then inverted with

    IndexError: index 10 is out of bounds for axis 0 with size 3

and string labels died inside `int()`. Canonical 0/1 labels worked, which
is why it survived (#146). `Blocks` and `Nested` were already fine --
neither indexes anything by the label -- so this is the tree alone.

The fixture matters more than usual here. A tree whose leaf clusters hang
off the same parent gives every leaf the same ancestor variance, so
relabelling cannot change the answer and the equalities below would hold
against a broken implementation. These leaves hang at UNEQUAL depth, and
the tests check that the strengths move the race and that a different
PARTITION gives a different race before any equality is believed.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities
from winning.factor.structures import Blocks, Nested, Tree

MU = np.array([-0.5, -0.1, 0.2, 0.4])
D = np.array([0.8, 0.9, 1.0, 1.1])
V = np.array([0.2, 0.3, -0.1, 0.4])
POINTS = 257

# leaves 0 and 1 hang off node 3, leaf 2 off node 4: unequal depth, so
# the ancestor variance differs by cluster and a relabelling is visible.
TREE = dict(parent=[3, 3, 4, 4, -1], strength=[0.0, 0.0, 0.0, 0.6, 0.4])
FLAT = dict(parent=[3, 3, 4, 4, -1], strength=[0.0] * 5)

SPELLINGS = {
    "canonical": [0, 0, 1, 2],
    "arbitrary ints": [10, 10, 20, 30],
    "strings": ["a", "a", "b", "c"],
    "negatives": [-5, -5, 0, 7],
}


def _fwd(cluster, **kw):
    return np.asarray(race_probabilities(
        MU, structure=Tree(cluster, V, D, **(kw or TREE)), points=POINTS))


def test_the_tree_strengths_move_the_race():
    """Otherwise every equality below holds for a broken implementation."""
    assert np.abs(_fwd([0, 0, 1, 2]) - _fwd([0, 0, 1, 2], **FLAT)).max() > 1e-3


def test_a_different_partition_is_a_different_race():
    """The other half: the labels must carry information."""
    assert np.abs(_fwd([0, 0, 1, 2]) - _fwd([0, 0, 2, 1])).max() > 1e-3


@pytest.mark.parametrize("name", list(SPELLINGS))
def test_the_forward_reads_every_label_spelling_alike(name):
    got = _fwd(SPELLINGS[name])
    assert np.abs(got - _fwd(SPELLINGS["canonical"])).max() < 1e-14


@pytest.mark.parametrize("name", list(SPELLINGS))
def test_the_inverse_reads_every_label_spelling_alike(name):
    s = Tree(SPELLINGS[name], V, D, **TREE)
    p = _fwd(SPELLINGS["canonical"])
    got = np.asarray(abilities_from_race(p, structure=s, points=POINTS))
    ref = np.asarray(abilities_from_race(
        p, structure=Tree(SPELLINGS["canonical"], V, D, **TREE),
        points=POINTS))
    assert np.abs(got - ref).max() < 1e-14


@pytest.mark.parametrize("name", list(SPELLINGS))
def test_the_tree_round_trips_under_every_spelling(name):
    s = Tree(SPELLINGS[name], V, D, **TREE)
    p = np.asarray(race_probabilities(MU, structure=s, points=POINTS))
    mu_hat = abilities_from_race(p, structure=s, points=POINTS)
    back = np.asarray(race_probabilities(mu_hat, structure=s, points=POINTS))
    assert np.abs(back - p).max() < 1e-7


@pytest.mark.parametrize("name", list(SPELLINGS))
def test_blocks_and_nested_were_already_label_agnostic(name):
    """Recorded so a future change cannot quietly break what worked."""
    cl = SPELLINGS[name]
    for s in (Blocks(cl, V, D),
              Nested(cl, V, D, coupling=np.full((4, 1), 0.2), gamma=0.3)):
        p = np.asarray(race_probabilities(MU, structure=s, points=POINTS))
        assert np.isfinite(p).all()
        abilities_from_race(p, structure=s, points=POINTS)
