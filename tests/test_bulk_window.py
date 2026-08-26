"""Thorough tests for the winner-bulk lattice window (now the default)."""
import numpy as np
import pytest
from scipy.stats import norm

from winning.factor.races import (race_probabilities, abilities_from_race,
                                  removal_shares, tie_densities)


def _field(seed=0, n=40, n_hopeless=25, spread=6.0):
    rng = np.random.default_rng(seed)
    n_live = n - n_hopeless
    mu = np.r_[rng.normal(0, 0.8, n_live), 2.0 + spread * rng.random(n_hopeless)]
    mu -= mu.mean()
    D = (0.5 + 0.5 * rng.random(n)) ** 2
    v = rng.normal(0, 0.6, n)
    return mu, v[:, None], D


# ------------------------------------------------- agreement with old window
@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_bulk_matches_span_at_high_resolution(base):
    mu, V, D = _field()
    p_bulk = race_probabilities(mu, V=V, D=D, base=base, points=2001)
    p_span = race_probabilities(mu, V=V, D=D, base=base, points=4001,
                                window="span")
    assert 0.5 * np.abs(p_bulk - p_span).sum() < 2e-6


def test_bulk_dominates_span_at_equal_points_on_hopeless_fields():
    mu, V, D = _field(seed=3, n=120, n_hopeless=95, spread=8.0)
    ref = race_probabilities(mu, V=V, D=D, points=4001, window="span")
    e_bulk = 0.5 * np.abs(race_probabilities(mu, V=V, D=D, points=65) - ref).sum()
    e_span = 0.5 * np.abs(race_probabilities(mu, V=V, D=D, points=65,
                                             window="span") - ref).sum()
    assert e_bulk < e_span / 100.0        # same budget, >=100x more accurate
    assert e_bulk < 1e-8


# ---------------------------------------------------- closed-form anchors
def test_two_runner_closed_form_at_33_points():
    mu = np.array([0.4, -0.4]); D = np.array([0.6, 1.1]); v = np.array([0.8, 0.3])
    p = race_probabilities(mu, V=v[:, None], D=D, points=33)
    exact = norm.cdf((mu[1] - mu[0]) / np.sqrt(D.sum() + (v[0] - v[1]) ** 2))
    assert abs(p[0] - exact) < 5e-5


def test_exchangeable_at_33_points():
    n = 10
    p = race_probabilities(np.zeros(n), V=None, D=np.ones(n), points=33)
    assert np.abs(p - 1.0 / n).max() < 1e-6


def test_gumbel_softmin_survives_bulk_window():
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 1.0, 10); mu -= mu.mean()
    p = race_probabilities(mu, V=None, D=np.full(10, np.pi ** 2 / 6),
                           base="gumbel", points=201)
    soft = np.exp(-mu) / np.exp(-mu).sum()
    assert 0.5 * np.abs(p - soft).sum() < 1e-4


# ----------------------------------------------------------- delta control
def test_delta_can_never_hurt():
    # the 2 sd safety pad dominates the delta truncation, so every delta is
    # effectively lossless -- assert the guarantee rather than a monotone
    # dial that the pad has (safely) flattened
    mu, V, D = _field(seed=5)
    ref = race_probabilities(mu, V=V, D=D, points=4001, delta=1e-14)
    for d in (1e-4, 1e-8, 1e-12):
        e = 0.5 * np.abs(race_probabilities(mu, V=V, D=D, points=801,
                                            delta=d) - ref).sum()
        assert e < 1e-10


# ------------------------------------------------------- downstream verbs
def test_slopes_consistent_across_windows():
    mu, V, D = _field(seed=7, n=20, n_hopeless=8)
    _, s_bulk = race_probabilities(mu, V=V, D=D, points=2001,
                                   return_slopes=True)
    _, s_span = race_probabilities(mu, V=V, D=D, points=4001, window="span",
                                   return_slopes=True)
    assert np.abs(s_bulk - s_span).max() < 1e-4


def test_inversion_round_trip_fast_lattice():
    mu, V, D = _field(seed=9, n=25, n_hopeless=10, spread=3.0)
    p = race_probabilities(mu, V=V, D=D, points=1001)
    mu_hat = abilities_from_race(p, V=V, D=D, points=101)
    p_back = race_probabilities(mu_hat, V=V, D=D, points=1001)
    assert 0.5 * np.abs(p_back - p).sum() < 5e-4


def test_removal_and_ties_still_consistent():
    mu, V, D = _field(seed=11, n=15, n_hopeless=5, spread=3.0)
    Q = np.asarray(removal_shares(mu, V=V, D=D, points=1001))
    keep = np.arange(len(mu)) != 2
    direct = race_probabilities(mu[keep], V=V[keep], D=D[keep], points=1001)
    row = Q[2, keep] / Q[2, keep].sum()
    assert 0.5 * np.abs(row - direct).sum() < 2e-3
    T = np.asarray(tie_densities(mu, V=V, D=D, points=1001))
    assert np.all(T[~np.eye(len(mu), dtype=bool)] >= 0)


# ------------------------------------------------------------- edge cases
def test_extreme_heteroskedasticity():
    rng = np.random.default_rng(13)
    n = 12
    mu = rng.normal(0, 1, n); mu -= mu.mean()
    D = np.exp(rng.uniform(-4, 4, n))          # 3000:1 variance ratios
    p = race_probabilities(mu, V=None, D=D, points=201)
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-9
    ref = race_probabilities(mu, V=None, D=D, points=4001, window="span")
    assert 0.5 * np.abs(p - ref).sum() < 1e-4


def test_one_dominant_runner():
    mu = np.array([-6.0] + [0.5] * 9); mu -= mu.mean()
    p = race_probabilities(mu, V=None, D=np.ones(10), points=101)
    assert p[0] > 0.9999
    assert np.all(p[1:] > 0)                   # smooth positives, never zero
