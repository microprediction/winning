"""The compiled kernels reach the ratings hot path, and the one switch
controls them.

Found 2026-09-21: `rust_active()` returned True while `update_winner_full`
ran pure numpy for 97% of its time, because `factor.core` -- whose two
kernels ARE that 97% -- was not in `rustconfig._rust_modules()`, and
`methods.native` checked `_fastrace is not None` directly, ignoring
`WINNING_PURE` and `use_rust(False)`. The user-facing promise is
`pip install winning[fast]` makes it fast; a hand-kept module list is
how that promise silently broke.
"""
import pathlib
import re

import numpy as np
import pytest

import winning
from winning import rustconfig
from winning.factor import core
from winning.factor.core import (hermite_nodes, jacobian_vector_product,
                                 win_probabilities_factor)

fastrace = pytest.importorskip("fastrace")

K_RANK = [(8, 1), (8, 2), (20, 1), (20, 2)]


def _field(K, r, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=K)
    V = rng.normal(size=(K, r)) * 0.5
    D = 0.6 + rng.random(K)
    F, W = hermite_nodes(r, Q=15)
    h = rng.normal(size=K)
    return mu, V, D, F, W, h


@pytest.fixture
def rust_on_after():
    yield
    winning.use_rust(True)


@pytest.mark.parametrize("K,r", K_RANK)
def test_forward_kernel_rust_matches_numpy(K, r, rust_on_after):
    mu, V, D, F, W, _ = _field(K, r)
    winning.use_rust(True)
    assert core._HAVE_RUST
    p_rust, tot_rust = win_probabilities_factor(mu, V, D, F, W, return_total=True)
    winning.use_rust(False)
    assert not core._HAVE_RUST
    p_np, tot_np = win_probabilities_factor(mu, V, D, F, W, return_total=True)
    assert np.abs(p_rust - p_np).max() < 1e-12
    assert abs(tot_rust - tot_np) < 1e-12


@pytest.mark.parametrize("K,r", K_RANK)
@pytest.mark.parametrize("form", ["ibp", "grid"])
def test_jvp_rust_matches_numpy(K, r, form, rust_on_after):
    mu, V, D, F, W, h = _field(K, r)
    winning.use_rust(True)
    j_rust = jacobian_vector_product(mu, V, D, F, W, h, form=form)
    winning.use_rust(False)
    j_np = jacobian_vector_product(mu, V, D, F, W, h, form=form)
    assert np.abs(j_rust - j_np).max() < 1e-12 * max(1.0, np.abs(j_np).max())


def test_numpy_only_options_fall_through(rust_on_after):
    """Deletions, per-node windows and the unnormalized JVP have no
    compiled form; asking for them must give the numpy answer, not raise."""
    mu, V, D, F, W, h = _field(8, 1)
    winning.use_rust(True)
    out, q = win_probabilities_factor(mu, V, D, F, W, return_deletions=True)
    assert q.shape == (8, 8)
    p = win_probabilities_factor(mu, V, D, F, W, per_node_interval=True)
    assert abs(p.sum() - 1) < 1e-9
    j = jacobian_vector_product(mu, V, D, F, W, h, normalized=False)
    assert j.shape == (8,)


def test_single_node_stays_on_numpy(rust_on_after):
    """Measured: at one factor node the compiled JVP is SLOWER (0.5 ms
    fixed cost) and the forward is a wash. nway's per-node loop makes 119
    such calls; dispatching them regressed it 25%. So one node is numpy."""
    mu, V, D, _, _, h = _field(8, 1)
    F1, W1 = np.zeros((1, 1)), np.ones(1)
    winning.use_rust(True)
    a = win_probabilities_factor(mu, V, D, F1, W1)
    b = jacobian_vector_product(mu, V, D, F1, W1, h)
    winning.use_rust(False)
    assert np.array_equal(a, win_probabilities_factor(mu, V, D, F1, W1))
    assert np.array_equal(b, jacobian_vector_product(mu, V, D, F1, W1, h))


def test_the_switch_reaches_core_and_native(rust_on_after):
    from winning.methods import native
    winning.use_rust(False)
    assert not core._HAVE_RUST and not native._HAVE_RUST
    assert not winning.rust_active()
    winning.use_rust(True)
    assert core._HAVE_RUST and native._HAVE_RUST
    assert winning.rust_active()


def test_every_module_that_imports_fastrace_is_on_the_switch():
    """The guard that would have caught core. A module that imports
    fastrace but is not registered has a private switch nobody can flip:
    `use_rust(False)` and WINNING_PURE do not reach it, and rust_active()
    lies about it."""
    root = pathlib.Path(winning.__file__).parent
    importers = set()
    for path in root.rglob("*.py"):
        if "research" in path.parts:
            continue
        if re.search(r"^\s*import fastrace\b", path.read_text(), re.M):
            rel = path.relative_to(root.parent).with_suffix("")
            importers.add(".".join(rel.parts))
    registered = {m.__name__ for m in rustconfig._rust_modules()}
    missing = sorted(importers - registered)
    assert not missing, (
        "modules that import fastrace but are not in "
        "rustconfig._rust_modules(), so use_rust()/WINNING_PURE cannot "
        "reach them:\n  " + "\n  ".join(missing))
