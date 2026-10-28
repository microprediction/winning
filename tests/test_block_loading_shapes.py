"""A rank-one cluster loading is the same race in every spelling.

`winning.shapes.as_loadings` is the ONE place that rule is decided: a
scalar, a length-n vector, (n, 1) and (1, n) all describe the same
rank-one loading. `block_race_probabilities` went through it and
`block_race_jacobian` did not -- it called `np.asarray` and kept whatever
shape it was handed, so the (n, 1) spelling reached the kernel as a
matrix and died inside an unrelated `np.take`:

    ValueError: input operand has more dimensions than allowed by the
    axis remapping

Every inverse and polishing path that calls the Jacobian inherited it, so
a caller who wrote `v[:, None]` got a forward pass that worked and an
inverse that crashed (#145).

This file follows tests/test_shape_contract.py: a DISCOVERY sweep so a
new verb cannot quietly skip the contract, NON-CONSTANT fixtures, and
every verb shown to MOVE the answer before any equality is believed.
"""
from __future__ import annotations

import numpy as np
import pytest

from winning.factor.blocks import (abilities_from_block_race,
                                   block_race_jacobian,
                                   block_race_probabilities)
from winning.factor.polish import polish_race
from winning.factor.structures import Blocks

MU = np.array([-0.5, -0.1, 0.2, 0.4])
D = np.array([0.8, 0.9, 1.0, 1.1])
CLUSTER = np.array([0, 0, 1, 1])
V = np.array([0.2, 0.3, -0.1, 0.4])          # non-constant on purpose
POINTS = 129


def _spellings():
    return {
        "vector (n,)": V,
        "matrix (n, 1)": V[:, None],
        "matrix (1, n)": V[None, :],
    }


def _p(loading):
    return block_race_probabilities(MU, CLUSTER, loading, D, points=POINTS)


def test_the_loading_actually_moves_the_answer():
    """Without this, every equality below could hold because the loading
    is ignored. A constant loading column is gauge-fixed to zero and
    cannot tell a broken block race from a correct independent one."""
    with_v = _p(V)
    without = _p(np.zeros_like(V))
    assert np.abs(with_v - without).max() > 1e-3, "the loading does nothing"
    J_with = block_race_jacobian(MU, CLUSTER, V, D, points=POINTS)
    J_without = block_race_jacobian(MU, CLUSTER, np.zeros_like(V), D,
                                    points=POINTS)
    assert np.abs(J_with - J_without).max() > 1e-3


@pytest.mark.parametrize("name", list(_spellings()))
def test_the_forward_reads_every_spelling_as_the_same_race(name):
    p = _p(_spellings()[name])
    assert np.abs(p - _p(V)).max() < 1e-14


@pytest.mark.parametrize("name", list(_spellings()))
def test_the_jacobian_reads_every_spelling_as_the_same_race(name):
    J = block_race_jacobian(MU, CLUSTER, _spellings()[name], D, points=POINTS)
    ref = block_race_jacobian(MU, CLUSTER, V, D, points=POINTS)
    assert J.shape == (len(MU), len(MU))
    assert np.abs(J - ref).max() < 1e-14
    assert np.abs(J.sum(axis=1)).max() < 1e-12      # rows sum to zero


@pytest.mark.parametrize("name", list(_spellings()))
def test_the_inverse_reads_every_spelling_as_the_same_race(name):
    p = _p(V)
    mu_hat = abilities_from_block_race(p, CLUSTER, _spellings()[name],
                                       D, points=POINTS)[0]
    assert np.abs(np.asarray(mu_hat) - (MU - MU.mean())).max() < 1e-10


@pytest.mark.parametrize("name", list(_spellings()))
def test_polishing_reads_every_spelling_as_the_same_race(name):
    res = polish_race(p0=_p(V), points=POINTS,
                      structure=Blocks(cluster=CLUSTER,
                                       loading=_spellings()[name], D=D))
    mu_hat = res.mu if hasattr(res, "mu") else res[1]
    assert np.abs(np.asarray(mu_hat) - (MU - MU.mean())).max() < 1e-10


def test_a_scalar_loading_is_the_same_for_every_entity():
    ref = block_race_jacobian(MU, CLUSTER, np.full(len(MU), 0.3), D,
                              points=POINTS)
    got = block_race_jacobian(MU, CLUSTER, 0.3, D, points=POINTS)
    assert np.abs(got - ref).max() < 1e-14


def test_rank_two_is_still_refused_with_its_own_reason():
    with pytest.raises(NotImplementedError, match="rank-one"):
        block_race_jacobian(MU, CLUSTER, np.ones((len(MU), 2)), D,
                            points=POINTS)


def test_the_jacobian_keeps_its_docstring():
    """It had none: a statement sat above the string literal, so python
    never bound it as __doc__ and help() showed nothing."""
    assert block_race_jacobian.__doc__
    assert "min-wins" in block_race_jacobian.__doc__
