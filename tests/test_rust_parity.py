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
