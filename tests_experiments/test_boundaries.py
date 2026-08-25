"""Tests for exp14: generic-base factor race anchors and calibrations."""

import sys
from pathlib import Path

import numpy as np

for sub in ("", "exp13_ghk_benchmark", "exp14_boundaries"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments" / sub))
import run_boundaries as bd  # noqa: E402
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

RNG = np.random.default_rng(5)


def test_gumbel_base_zero_loadings_is_exact_softmax():
    # standardized (unit-variance) Gumbel base: Luce with inverse scale pi/sqrt(6)
    mu = RNG.normal(0, 0.9, 15)
    F, W = hermite_nodes(1)
    p, _ = bd.factor_shares_base(mu, np.zeros((15, 1)), np.ones(15), F, W, base="gumbel")
    soft = np.exp(-mu * np.pi / np.sqrt(6.0)); soft /= soft.sum()
    assert np.abs(p - soft).max() < 1e-12


def test_normal_base_matches_raceutil():
    n = 12
    mu = RNG.normal(0, 0.7, n)
    V = 0.4 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.5, 1.4, n)
    F, W = hermite_nodes(2)
    p1, _ = bd.factor_shares_base(mu, V, D, F, W, base="normal")
    p2 = win_probabilities_factor(mu, V, D, F, W)
    assert np.abs(p1 - p2).max() < 5e-5


def test_slopes_are_negative_and_match_finite_differences():
    n = 8
    mu = RNG.normal(0, 0.5, n)
    V = 0.3 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.6, 1.2, n)
    F, W = hermite_nodes(2)
    for base in ("normal", "gumbel"):
        p, sl = bd.factor_shares_base(mu, V, D, F, W, base=base)
        assert np.all(sl < 0)
        eps = 1e-4
        for i in (0, 3):
            mp = mu.copy(); mp[i] += eps
            pp, _ = bd.factor_shares_base(mp, V, D, F, W, base=base)
            fd = (pp[i] - p[i]) / eps
            # slope returned is normalized by total mass, matches unnormalized fd
            assert abs(fd - sl[i]) < 0.05 * max(abs(sl[i]), 1e-3)


def test_calibration_roundtrip_both_bases():
    n = 20
    mu_true = RNG.normal(0, 0.6, n); mu_true -= mu_true.mean()
    V = 0.4 * RNG.standard_normal((n, 2))
    D = RNG.uniform(0.5, 1.4, n)
    F, W = hermite_nodes(2)
    for base in ("normal", "gumbel"):
        target, _ = bd.factor_shares_base(mu_true, V, D, F, W, base=base)
        mu_hat = bd.calibrate_base(target, V, D, F, W, base=base)
        back, _ = bd.factor_shares_base(mu_hat, V, D, F, W, base=base)
        assert np.abs(back - target).max() < 1e-6


def test_spectral_corr_is_valid_correlation():
    basis, _ = np.linalg.qr(RNG.standard_normal((20, 20)))
    C, eig = bd.spectral_corr(20, 1.5, basis)
    assert np.abs(np.diag(C) - 1.0).max() < 1e-12
    assert np.linalg.eigvalsh(C).min() > -1e-10
    assert eig[0] >= eig[-1] > 0          # disclosed post-rescale spectrum
    # shared basis: same basis, different gamma -> comparisons not confounded
    C2, _ = bd.spectral_corr(20, 3.0, basis)
    assert not np.allclose(C, C2)
