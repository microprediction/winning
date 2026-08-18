"""Tests for correlated races (thurstone.correlated).

Conventions under test: min wins, lower ability = stronger, abilities in
physical units.
"""

import numpy as np
import pytest

from winning.thurstone import (
    Density,
    FactorRace,
    Race,
    UniformLattice,
    factor_model,
    gaussian_factor_race,
    hermite_nodes,
    solve_abilities,
)
from winning.thurstone.inference import densities_from_offsets

LAT = UniformLattice(L=400, unit=0.05)
RNG = np.random.default_rng(7)


def circle_kernel(n: int, ell: float) -> np.ndarray:
    th = 2 * np.pi * np.arange(n) / n
    d = np.abs((th[:, None] - th[None, :] + np.pi) % (2 * np.pi) - np.pi)
    return np.exp(-d / ell)


def mc_reference(mu, C, scale, n_draws, seed=9):
    """Monte Carlo win frequencies for a correlated Gaussian race (min wins)."""
    L = np.linalg.cholesky(C + 1e-9 * np.eye(len(C)))
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(mu))
    done = 0
    while done < n_draws:
        n = min(200_000, n_draws - done)
        X = np.asarray(mu)[:, None] + scale * (L @ rng.standard_normal((len(mu), n)))
        counts += np.bincount(np.argmin(X, axis=0), minlength=len(mu))
        done += n
    return counts / counts.sum()


def test_independent_limit_matches_race():
    """Zero loadings must reproduce the package's independent state prices."""
    mu = np.array([-0.6, -0.2, 0.0, 0.3, 0.7])
    base = Density.skew_normal(LAT, loc=0.0, scale=1.0, a=0.0)
    fr = FactorRace(base, mu, np.zeros((5, 1)))
    p_fr = fr.state_prices()
    dens = densities_from_offsets(base, list(mu / LAT.unit))  # offsets in steps
    p_race = Race(dens).state_prices()
    assert np.abs(p_fr - p_race).max() < 2e-3


def test_known_factor_model_matches_monte_carlo():
    """With Sigma = V V^T + diag(D) known exactly, quadrature matches MC."""
    n, k = 8, 2
    V = 0.5 * RNG.standard_normal((n, k))
    D = RNG.uniform(0.4, 0.9, n)
    C = V @ V.T + np.diag(D)
    mu = RNG.normal(0.0, 0.5, n)
    bases = [Density.skew_normal(LAT, 0.0, float(np.sqrt(d)), 0.0) for d in D]
    p = FactorRace(bases, mu, V).state_prices()
    ref = mc_reference(mu, C, 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 4e-3


def test_equicorrelated_single_factor_exact():
    """Equicorrelation is exactly one factor; k=1 must already be exact."""
    n, rho = 10, 0.5
    C = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    mu = RNG.normal(0.0, 0.5, n)
    p = gaussian_factor_race(LAT, C, 1, mu).state_prices()
    ref = mc_reference(mu, C, 1.0, 2_000_000)
    assert np.abs(p - ref).max() < 4e-3


def test_factor_model_identity_gives_no_correlation():
    """Factor analysis of C = I must not invent off-diagonal correlation."""
    V, D = factor_model(np.eye(12), 3)
    C_hat = V @ V.T + np.diag(D)
    off = C_hat - np.diag(np.diag(C_hat))
    assert np.abs(off).max() < 1e-6


def test_gumbel_min_independent_race_is_softmax():
    """Independent Gumbel-min race = Luce/softmax(-mu/scale), exactly."""
    mu = np.array([-0.8, -0.3, 0.0, 0.4, 1.0])
    scale = 0.7
    base = Density.gumbel_min(LAT, loc=0.0, scale=scale)
    p = FactorRace(base, mu, np.zeros((5, 1))).state_prices()
    z = -mu / scale
    softmax = np.exp(z - z.max())
    softmax /= softmax.sum()
    assert np.abs(p - softmax).max() < 2e-3


def test_correlated_softmax_departs_from_luce_and_sums_to_one():
    """Nonzero loadings on a Gumbel base: a non-IIA softmax generalization."""
    mu = np.array([-0.5, -0.5, 0.2, 0.2, 0.6, 0.6])
    base = Density.gumbel_min(LAT, loc=0.0, scale=0.7)
    V = np.zeros((6, 1))
    V[:2, 0] = 0.8  # the first two share an environment
    p = FactorRace(base, mu, V).state_prices()
    z = -mu / 0.7
    softmax = np.exp(z - z.max())
    softmax /= softmax.sum()
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.abs(p - softmax).max() > 5e-3  # correlation must matter


def test_deletion_ensemble_matches_per_scratch_recompute():
    n = 6
    C = circle_kernel(n, 1.0)
    mu = RNG.normal(0.0, 0.5, n)
    fr = gaussian_factor_race(LAT, C, 2, mu)
    q = fr.deletion_ensemble()
    assert np.allclose(q.sum(axis=1), 1.0)
    for i in (0, 3):
        keep = np.setdiff1d(np.arange(n), [i])
        direct = fr.state_prices(keep=keep)
        assert np.abs(direct - q[i][keep]).max() < 1e-10


def test_scratch_after_calibration_favors_correlated_partner():
    """The neighbor-inheritance effect runs through the inverse map: correlated
    partners cannibalize each other, so matching equal observed win frequencies
    forces them to stronger fitted abilities; scratching one then hands its wins
    disproportionately to the partner.

    (At EQUAL abilities the effect vanishes: scratching is a marginal, and the
    survivors here are mutually independent, so redistribution is exactly
    uniform -- the deletion-semantics point.)"""
    V = np.array([[0.9], [0.9], [0.0], [0.0]])
    bases = [
        Density.skew_normal(LAT, 0.0, float(np.sqrt(max(1 - v[0] ** 2, 1e-3))), 0.0) for v in V
    ]
    # equal-ability sanity: uniform redistribution
    q0 = FactorRace(bases, np.zeros(4), V).deletion_ensemble()
    assert np.abs(q0[0] - np.array([0.0, 1 / 3, 1 / 3, 1 / 3])).max() < 1e-3

    # calibrate to equal observed frequencies, then scratch
    mu_fit = solve_abilities(bases, V, np.full(4, 0.25), n_iter=400)
    assert mu_fit[0] < mu_fit[2]  # partners fitted stronger (min wins)
    q = FactorRace(bases, mu_fit, V).deletion_ensemble()
    assert q[0][1] > q[0][2] + 5e-3
    assert q[0][1] > q[0][3] + 5e-3


def test_solve_abilities_roundtrip_under_correlation():
    n = 8
    C = circle_kernel(n, 1.2)
    mu_true = RNG.normal(0.0, 0.4, n)
    mu_true -= mu_true.mean()
    fr = gaussian_factor_race(LAT, C, 2, mu_true)
    target = fr.state_prices()
    V, D = factor_model(C, 2)
    bases = [Density.skew_normal(LAT, 0.0, float(np.sqrt(d)), 0.0) for d in D]
    mu_fit = solve_abilities(bases, V, target, n_iter=400)
    back = FactorRace(bases, mu_fit, V).state_prices()
    assert np.abs(back - target).max() < 2e-3


def test_hermite_nodes_integrate_gaussian_moments():
    # tolerances reflect node pruning (corner nodes carry x^2-weighted mass)
    F, W = hermite_nodes(2, Q=15)
    assert abs(W.sum() - 1.0) < 5e-6
    assert np.abs(W @ F).max() < 5e-6  # mean zero
    assert np.abs(W @ (F**2) - 1.0).max() < 5e-6  # unit variance


@pytest.mark.parametrize("keep", [[0, 2, 4], [1, 3]])
def test_state_prices_keep_subsets_sum_to_one(keep):
    mu = RNG.normal(0.0, 0.5, 5)
    fr = gaussian_factor_race(LAT, circle_kernel(5, 1.0), 2, mu)
    p = fr.state_prices(keep=keep)
    assert len(p) == len(keep)
    assert abs(p.sum() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Edge cases, error handling, and structural properties
# ---------------------------------------------------------------------------


def test_shared_base_equals_list_of_identical_bases():
    mu = RNG.normal(0.0, 0.5, 4)
    V = 0.4 * RNG.standard_normal((4, 2))
    base = Density.skew_normal(LAT, 0.0, 0.8, 0.0)
    p_shared = FactorRace(base, mu, V).state_prices()
    p_list = FactorRace([base] * 4, mu, V).state_prices()
    assert np.abs(p_shared - p_list).max() < 1e-14


def test_ability_translation_invariance():
    """Adding a constant to every ability must not change prices."""
    mu = RNG.normal(0.0, 0.5, 5)
    V = 0.4 * RNG.standard_normal((5, 2))
    base = Density.skew_normal(LAT, 0.0, 0.8, 0.0)
    p1 = FactorRace(base, mu, V).state_prices()
    p2 = FactorRace(base, mu + 0.7, V).state_prices()
    assert np.abs(p1 - p2).max() < 1e-6


def test_stronger_ability_wins_more():
    """Min wins: lowering an ability must raise that competitor's price."""
    mu = np.zeros(4)
    base = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    V = 0.3 * np.ones((4, 1))
    p0 = FactorRace(base, mu, V).state_prices()
    mu2 = mu.copy()
    mu2[1] -= 0.5
    p2 = FactorRace(base, mu2, V).state_prices()
    assert p2[1] > p0[1] + 0.02


def test_prices_always_sum_to_one_and_are_positive():
    for trial in range(5):
        n = int(RNG.integers(2, 9))
        k = int(RNG.integers(1, 4))
        mu = RNG.normal(0.0, 0.7, n)
        V = 0.5 * RNG.standard_normal((n, k))
        base = Density.skew_normal(LAT, 0.0, 0.9, 0.0)
        p = FactorRace(base, mu, V).state_prices()
        assert abs(p.sum() - 1.0) < 1e-12
        assert np.all(p > 0)


def test_mismatched_loadings_rows_raises():
    base = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="one row per competitor"):
        FactorRace(base, np.zeros(3), np.zeros((4, 1)))


def test_mismatched_base_count_raises():
    base = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="one base density per competitor"):
        FactorRace([base, base], np.zeros(3), np.zeros((3, 1)))


def test_mismatched_lattices_raises():
    b1 = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    b2 = Density.skew_normal(UniformLattice(L=200, unit=0.05), 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="same lattice"):
        FactorRace([b1, b2], np.zeros(2), np.zeros((2, 1)))


def test_solve_abilities_rejects_nonpositive_targets():
    base = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        solve_abilities(base, np.zeros((3, 1)), [0.5, 0.5, 0.0])


def test_factor_model_matches_diagonal_exactly():
    C = circle_kernel(9, 0.9)
    V, D = factor_model(C, 3)
    C_hat = V @ V.T + np.diag(D)
    assert np.abs(np.diag(C_hat) - np.diag(C)).max() < 1e-9
    assert np.all(D > 0)


def test_factor_model_equicorrelated_is_exact_at_k1():
    n, rho = 7, 0.35
    C = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    V, D = factor_model(C, 1)
    assert np.abs(V @ V.T + np.diag(D) - C).max() < 1e-8


def test_gaussian_nodes_deterministic_given_seed():
    from winning.thurstone import gaussian_nodes
    F1, W1 = gaussian_nodes(5, n=512, seed=3)
    F2, W2 = gaussian_nodes(5, n=512, seed=3)
    assert np.array_equal(F1, F2) and np.array_equal(W1, W2)
    assert abs(W1.sum() - 1.0) < 1e-12


def test_gumbel_min_density_is_normalized_and_left_skewed():
    d = Density.gumbel_min(LAT, loc=0.0, scale=0.6)
    assert abs(d.p.sum() - 1.0) < 1e-9
    g = d.lattice.grid
    mean = float(d.p @ g)
    third = float(d.p @ (g - mean) ** 3)
    assert third < 0  # min-Gumbel has a long left tail


def test_correlated_softmax_iia_violation_has_right_sign():
    """Scratch one of two calibrated environment-sharers: the partner should gain
    MORE than the Luce renormalization predicts."""
    base = Density.gumbel_min(LAT, loc=0.0, scale=0.7)
    V = np.array([[0.8], [0.8], [0.0], [0.0]])
    mu = solve_abilities([base] * 4, V, np.full(4, 0.25), n_iter=300)
    fr = FactorRace([base] * 4, mu, V)
    p = fr.state_prices()
    q = fr.state_prices(keep=[1, 2, 3])  # scratch 0
    luce = p[1:] / p[1:].sum()  # IIA renormalization
    assert q[0] > luce[0] + 1e-3  # partner (1) gains extra


def test_deletion_ensemble_rows_exclude_the_scratched():
    mu = RNG.normal(0.0, 0.4, 5)
    fr = gaussian_factor_race(LAT, circle_kernel(5, 1.0), 2, mu)
    q = fr.deletion_ensemble()
    assert np.abs(np.diag(q)).max() == 0.0


def test_single_factor_race_two_competitors_symmetry():
    base = Density.skew_normal(LAT, 0.0, 1.0, 0.0)
    p = FactorRace(base, np.zeros(2), np.array([[0.5], [0.5]])).state_prices()
    assert np.abs(p - 0.5).max() < 1e-9
