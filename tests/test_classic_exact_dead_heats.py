"""The classic lattice prices dead heats exactly (#418, #362, #348, #373, #369).

Every expectation here is either a closed form (exchangeability, a
certain winner) or a brute-force enumeration of all joint atom outcomes
with a tied minimum split equally. Runs the pure path, which is what CI
exercises; the compiled kernel is pinned to it by test_rust_parity.
"""
import itertools

import numpy as np
import pytest

from winning import rustconfig
from winning.classic.lattice import (_exact_offset_cdfs, cdf_to_pdf,
                                     densities_from_offsets, integer_shift,
                                     pdf_to_cdf, skew_normal_density,
                                     state_prices_from_densities,
                                     state_prices_from_offsets)
from winning.classic.lattice_calibration import solve_for_implied_offsets


@pytest.fixture(autouse=True)
def _pure():
    was = rustconfig.rust_active()
    rustconfig.use_rust(False)
    yield
    rustconfig.use_rust(was)


def enumerate_prices(pdfs):
    """Exact equal-split winner claims by enumerating every outcome."""
    supports = [np.flatnonzero(np.asarray(p) > 0) for p in pdfs]
    out = np.zeros(len(pdfs))
    for combo in itertools.product(*supports):
        pr = np.prod([pdfs[i][t] for i, t in enumerate(combo)])
        lo = min(combo)
        winners = [i for i, t in enumerate(combo) if t == lo]
        for i in winners:
            out[i] += pr / len(winners)
    return out


def enumerate_offsets(density, offsets):
    _, cdfs, _ = _exact_offset_cdfs(density, offsets)
    return enumerate_prices([cdf_to_pdf(c) for c in cdfs])


def three_atom(n=17, at=7, w=(0.2, 0.3, 0.5)):
    d = np.zeros(n)
    d[at:at + len(w)] = w
    return d


# --- #418: two runners, compact support ----------------------------------

def test_two_runner_compact_support_is_exact():
    p = state_prices_from_offsets(three_atom(), [-1, 1])
    np.testing.assert_allclose(p, [0.95, 0.05], atol=1e-14)


def test_disjoint_support_pays_the_certain_winner():
    p = state_prices_from_offsets(three_atom(), [-2, 2])
    np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-14)


@pytest.mark.parametrize("seed", range(6))
def test_random_atom_laws_match_enumeration(seed):
    rng = np.random.default_rng(seed)
    d = np.zeros(15)
    d[5:10] = rng.dirichlet(np.ones(5))
    d[rng.integers(5, 10)] = 0.0       # a gap in the support
    d /= d.sum()
    n = int(rng.integers(2, 5))
    offsets = list(rng.integers(-3, 4, size=n).astype(float))
    p = state_prices_from_offsets(d, offsets)
    np.testing.assert_allclose(p, enumerate_offsets(d, offsets), atol=1e-13)
    assert abs(sum(p) - 1.0) < 1e-13


def test_general_densities_route_is_exact_too():
    d = three_atom()
    dens = [cdf_to_pdf(integer_shift(pdf_to_cdf(d), k)) for k in (-1, 1)]
    np.testing.assert_allclose(state_prices_from_densities(dens),
                               [0.95, 0.05], atol=1e-14)


# --- #362: random multiway ties ------------------------------------------

def test_twenty_iid_entrants_split_the_claim_equally():
    d = np.array([0, 0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0])
    p = state_prices_from_offsets(d, [0.0] * 20)
    np.testing.assert_allclose(p, np.full(20, 0.05), atol=1e-14)


def test_unequal_eight_runner_field_matches_enumeration():
    d = np.array([0, 0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0])
    offsets = [0, -1, 1, 0, 0, 0, 2, 0]
    want = [0.0918114072, 0.5380292620, 0.0029137021, 0.0918114072,
            0.0918114072, 0.0918114072, 0.0, 0.0918114072]
    p = state_prices_from_offsets(d, offsets)
    np.testing.assert_allclose(p, want, atol=1e-10)
    np.testing.assert_allclose(p, enumerate_offsets(d, offsets), atol=1e-14)


def test_point_mass_control():
    d = np.zeros(11)
    d[5] = 1.0
    np.testing.assert_allclose(state_prices_from_offsets(d, [0.0] * 5),
                               np.full(5, 0.2), atol=1e-15)


# --- #348: fractional offsets are priced as the mixture they are ---------

def _narrow(L):
    x = np.arange(2 * L + 1) - L
    d = np.exp(-0.5 * x ** 2.0)
    return d / d.sum()


@pytest.mark.parametrize("offsets", [[0.5, 0.5], [0.5, 1.5],
                                     [-2.5, 0.5, 3.5], [0.25, -0.75, 0.5]])
def test_fractional_offsets_are_exhaustive_and_exact(offsets):
    d = _narrow(41)
    p = state_prices_from_offsets(d, offsets)
    assert abs(sum(p) - 1.0) < 1e-13
    np.testing.assert_allclose(p, enumerate_offsets(d, offsets), atol=1e-13)


def test_identical_fractional_runners_are_exchangeable():
    p = state_prices_from_offsets(_narrow(41), [0.5, 0.5])
    np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-15)


def test_half_step_does_not_underpay_a_certain_winner():
    p = state_prices_from_offsets(_narrow(20), [15, 2.5])
    assert p[1] > 1 - 1e-12 and p[0] < 1e-15


# --- #373: a translated runner keeps its mass ----------------------------

@pytest.mark.parametrize("k", [0, 10, 19, 39, -19, -39, 19.5])
def test_singleton_and_pair_under_a_common_shift(k):
    d = np.ones(83) / 83
    assert state_prices_from_offsets(d, [k])[0] == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(state_prices_from_offsets(d, [k, k]),
                               [0.5, 0.5], atol=1e-14)


def test_common_integer_shift_leaves_an_unequal_field_unchanged():
    d = np.ones(83) / 83          # broad: mass right up to both edges
    base = np.array([-5.0, 0.0, 3.0, 0.5])
    ref = state_prices_from_offsets(d, list(base))
    for c in (-30, -10, 10, 30):
        np.testing.assert_allclose(state_prices_from_offsets(d, list(base + c)),
                                   ref, atol=1e-13)


def test_integer_shift_keeps_the_cdf_endpoint():
    cdf = pdf_to_cdf(np.ones(83) / 83)
    for k in (-19, -1, 1, 19, 40):
        s = integer_shift(cdf, k)
        assert s[-1] == cdf[-1]
        assert np.all(np.diff(s) >= 0)
    for f in densities_from_offsets(np.ones(83) / 83, [19.0, -19.0, 3.5]):
        assert abs(sum(f) - 1.0) < 1e-13


# --- the inverse converges to the exact forward --------------------------

@pytest.mark.parametrize("density, offsets", [
    (three_atom(), [-1.0, 1.0]),
    (three_atom(), [-1.5, 0.0, 0.7, 2.0]),
    (_narrow(20), [0.0, 1.5, 3.0, -2.0]),
    (np.ones(83) / 83, [-5.0, 0.0, 3.0, 10.0]),
])
def test_inverse_round_trips_compact_laws(density, offsets):
    target = state_prices_from_offsets(density, offsets)
    a = solve_for_implied_offsets(target, density, nIter=10,
                                  implied_offsets_guess=[0.0] * len(offsets))
    back = state_prices_from_offsets(density, list(a))
    assert np.abs(np.asarray(back) - target).max() < 1e-6


# --- #369: the default guess has one offset per target -------------------

def test_default_guess_is_one_zero_per_price():
    density = skew_normal_density(15, 0.2, a=1.5)
    target = state_prices_from_offsets(density, [6, -2])
    a = solve_for_implied_offsets(target, density)
    b = solve_for_implied_offsets(target, density, implied_offsets_guess=[0, 0])
    np.testing.assert_allclose(a, b, atol=1e-15)
    back = state_prices_from_offsets(density, list(a))
    assert np.abs(np.asarray(back) - target).max() < 1e-4


def test_a_guess_of_the_wrong_length_is_refused():
    density = skew_normal_density(15, 0.2, a=1.5)
    with pytest.raises(ValueError, match="one starting offset per price"):
        solve_for_implied_offsets([0.5, 0.5], density,
                                  implied_offsets_guess=[0, 1, 2, 3, 4])


# --- big fields: node doubling keeps exactness without O(n^2 L) ---------

def test_big_atomic_field_still_splits_ties_exactly():
    d = np.array([0, 0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0])
    p = state_prices_from_offsets(d, [0.0] * 60)      # needs the exact 31 nodes
    np.testing.assert_allclose(p, np.full(60, 1 / 60), atol=1e-14)


def test_big_smooth_field_stops_doubling_early():
    from winning.classic import lattice as Lt
    x = np.arange(2001) - 1000
    d = np.exp(-0.5 * (x / 80.0) ** 2)
    d /= d.sum()
    offsets = list(np.linspace(-100, 100, 400))
    seen = []
    real = Lt._gauss_legendre01
    Lt._gauss_legendre01 = lambda q: (seen.append(q), real(q))[1]
    try:
        p = state_prices_from_offsets(d, offsets)
    finally:
        Lt._gauss_legendre01 = real
    assert abs(sum(p) - 1) < 1e-13
    assert max(seen) <= 32, seen          # the exact rule would be 201 nodes
