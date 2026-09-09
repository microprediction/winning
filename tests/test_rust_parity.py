"""Rust/python parity: every kernel with a compiled path must agree with the
numpy reference to numerical noise, because the numpy path is the spec.

Each test toggles the module-level _HAVE_RUST flag to force the reference
path and compares. Skipped wholesale if fastrace is not importable.
"""
import numpy as np
import pytest

fastrace = pytest.importorskip("fastrace")

import winning.factor.races as races
import winning.factor.blocks as blocks
import winning.classic.lattice as lattice
import winning.classic.lattice_calibration as lc
from winning.classic.lattice import skew_normal_density
from winning.factor.blocks import tree_race_probabilities

RNG = np.random.default_rng(7)


def _toggle(mod, value):
    old = mod._HAVE_RUST
    mod._HAVE_RUST = value
    return old


def test_factor_front_door_parity():
    n = 15
    mu = RNG.normal(size=n)
    V = RNG.normal(size=(n, 3)) * 0.4
    D = 0.4 + RNG.random(n)
    pr, sr = races.race_probabilities(mu, V=V, D=D, points=257,
                                      return_slopes=True)
    old = _toggle(races, False)
    try:
        pp, sp = races.race_probabilities(mu, V=V, D=D, points=257,
                                          return_slopes=True)
    finally:
        races._HAVE_RUST = old
    assert np.abs(pr - pp).max() < 1e-12
    assert np.abs(sr - sp).max() < 1e-12


def test_factor_independent_and_gumbel_fallback():
    mu = np.array([-0.5, 0.0, 0.8, 2.0])
    pr = races.race_probabilities(mu, points=257)
    old = _toggle(races, False)
    try:
        pp = races.race_probabilities(mu, points=257)
    finally:
        races._HAVE_RUST = old
    assert np.abs(pr - pp).max() < 1e-12
    # gumbel base must not dispatch to the normal-only kernel: with the
    # softmin scaling D = pi^2/6 the race IS softmax(-mu), rust flag or not
    D = np.full(len(mu), np.pi ** 2 / 6.0)
    pg = races.race_probabilities(mu, D=D, base="gumbel", points=1001)
    lu = np.exp(-mu) / np.exp(-mu).sum()
    assert np.abs(pg - lu).max() < 1e-6
    old = _toggle(races, False)
    try:
        pg2 = races.race_probabilities(mu, D=D, base="gumbel", points=1001)
    finally:
        races._HAVE_RUST = old
    # gumbel now has its OWN compiled path (forward_and_slopes_base);
    # the guard is that it is never priced as normal -- the softmax
    # identity above -- not bit-identity between backends
    assert np.abs(pg - pg2).max() < 1e-12


def test_tree_race_parity():
    n = 12
    mu = RNG.normal(size=n)
    D = 0.5 + RNG.random(n)
    cluster = np.repeat(np.arange(4), 3)
    loading = 0.2 + 0.3 * RNG.random(n)
    parent = np.array([4, 4, 5, 5, 6, 6, -1])
    strength = np.array([0.4, 0.3, 0.5, 0.35, 0.25, 0.2, 0.0])
    pr = tree_race_probabilities(mu, cluster, loading, D, parent, strength,
                                 points=257)
    old = _toggle(blocks, False)
    try:
        pp = tree_race_probabilities(mu, cluster, loading, D, parent,
                                     strength, points=257)
    finally:
        blocks._HAVE_RUST = old
    assert np.abs(pr - pp).max() < 1e-12


def test_classic_calibration_parity_and_roundtrip():
    density = skew_normal_density(L=500, unit=0.01, a=1.5)
    dividends = [2.0, 3.5, 6.0, 12.0, 20.0, 41.0]
    ar = lc.dividend_implied_ability(dividends, density)
    o1, o2 = _toggle(lattice, False), _toggle(lc, False)
    try:
        ap = lc.dividend_implied_ability(dividends, density)
    finally:
        lattice._HAVE_RUST, lc._HAVE_RUST = o1, o2
    assert np.abs(np.array(ar) - np.array(ap)).max() < 1e-10


def test_classic_state_prices_parity_with_ties_and_stragglers():
    density = skew_normal_density(L=400, unit=0.01, a=1.0)
    # exact ties (dead-heat multiplicity machinery) plus a near-hopeless
    # straggler exercise the epsilon conventions
    offsets = [-30.0, -30.0, 0.0, 55.5, 120.0]
    sr = lattice.state_prices_from_offsets(density, offsets)
    old = _toggle(lattice, False)
    try:
        sp = lattice.state_prices_from_offsets(density, offsets)
    finally:
        lattice._HAVE_RUST = old
    assert np.abs(np.array(sr) - np.array(sp)).max() < 1e-10
    # the tied pair must carry equal prices
    assert abs(sr[0] - sr[1]) < 1e-12


def test_forked_child_does_not_deadlock():
    """A forked child inherits the parent's warmed rayon pool, whose
    worker threads do not survive fork(2): before the with_usable_rayon
    guard, the child's first parallel region hung forever (measured as
    intermittent hangs in a fork-based consumer, 2026-09-05).
    The guard detects the foreign pid and installs a fresh local pool."""
    import multiprocessing as mp
    import os
    if os.name != "posix":
        pytest.skip("fork start method is POSIX-only")
    mu = RNG.normal(0, 1.2, 120)
    sd = np.ones(120)
    # warm the parent's global rayon pool through a parallel-branch call
    fastrace.top_k(mu, sd, 40, -12.0, 12.0, 513)
    ctx = mp.get_context("fork")
    p = ctx.Process(target=fastrace.top_k,
                    args=(mu, sd, 40, -12.0, 12.0, 513))
    p.start()
    p.join(timeout=30)
    alive = p.is_alive()
    if alive:
        p.terminate()
        p.join()
    assert not alive, "forked child deadlocked in a rayon parallel region"
    assert p.exitcode == 0


def test_rank_marginal_kernels_parity():
    """The compiled rank-marginal and rank-Jacobian kernels against the
    numpy path on identical windows, heteroscedastic field."""
    import winning.factor.topk as T
    if not (T._HAVE_RUST and hasattr(fastrace, "rank_marginals")
            and hasattr(fastrace, "rank_marginal_jacobian")):
        pytest.skip("fastrace without the rank kernels")
    n = 11
    mu = RNG.normal(size=n)
    D = 0.5 + RNG.random(n)
    P_r = T.rank_probabilities(mu, D=D)
    old = _toggle(T, False)
    try:
        P_p = T.rank_probabilities(mu, D=D)
    finally:
        T._HAVE_RUST = old
    assert np.abs(P_r - P_p).max() < 1e-12
    from winning.factor.races import BASES
    sd = np.sqrt(D)
    for r in (1, 2, 5, n):
        pr, jr = T._rank_marginal_with_jacobian(mu, sd, r, BASES["normal"],
                                                513, is_normal=True)
        pp, jp = T._rank_marginal_with_jacobian(mu, sd, r, BASES["normal"],
                                                513, is_normal=False)
        assert np.abs(pr - pp).max() < 1e-12, r
        assert np.abs(jr - jp).max() < 1e-12, r


def test_all_parity_scenarios_rust_matches_numpy():
    """The whole parity contract, Rust vs numpy in one process.

    Reuses parity/gen_vectors.build() -- the same 39 scenarios check.jl,
    check.mjs and check.R verify -- and runs it on both paths. Loops the
    contract so the Rust test cannot silently fall behind the others."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parity"))
    import gen_vectors as gv
    import winning

    winning.use_rust(True)
    if not winning.rust_active():
        pytest.skip("fastrace not active")
    rust = gv.build(gv.make_inputs())
    winning.use_rust(False)
    try:
        num = gv.build(gv.make_inputs())
    finally:
        winning.use_rust(True)

    bad = []
    for name in sorted(rust):
        vr = np.atleast_1d(np.asarray(rust[name]["value"], float)).ravel()
        vn = np.atleast_1d(np.asarray(num[name]["value"], float)).ravel()
        tol = float(rust[name].get("tol", 1e-9))
        d = float(np.abs(vr - vn).max()) if vr.size else 0.0
        if d > tol:
            bad.append(f"{name}: |rust-numpy|={d:.2e} > tol {tol:.0e}")
    assert len(rust) >= 39, f"only {len(rust)} scenarios built"
    assert not bad, "Rust != numpy on:\n  " + "\n  ".join(bad)
