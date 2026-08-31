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
    # absolute floor calibrated under the adaptive factor quadrature (more
    # nodes on sharp fields than the old fixed GH-15, slightly more
    # truncation accumulation at 65 points)
    assert e_bulk < 1e-7


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


# ---------------------------------------- hopeless runners are nearly free
@pytest.mark.parametrize("n_hopeless", [50, 500])
def test_adding_hopeless_runners_bounded_impact_factor(n_hopeless):
    """Adding irrelevant runners perturbs the live field by at most (a small
    multiple of) their total win mass -- the field-product bound, enforced."""
    rng = np.random.default_rng(21)
    n_live = 20
    mu = rng.normal(0, 0.8, n_live); mu -= mu.mean()
    D = (0.5 + 0.5 * rng.random(n_live)) ** 2
    v = rng.normal(0, 0.6, n_live)
    p_before = race_probabilities(mu, V=v[:, None], D=D, points=1001)
    mu_h = 4.0 + 4.0 * rng.random(n_hopeless)          # min-wins: hopeless
    mu2 = np.r_[mu, mu_h]
    D2 = np.r_[D, np.ones(n_hopeless)]
    v2 = np.r_[v, 0.3 * np.ones(n_hopeless)]
    p_after = race_probabilities(mu2, V=v2[:, None], D=D2, points=1001)
    mass = p_after[n_live:].sum()
    live = p_after[:n_live] / p_after[:n_live].sum()
    tv = 0.5 * np.abs(live - p_before).sum()
    # the additive term is engine resolution, not slack: the loadings are
    # gauge-centered, so adding runners shifts the column mean and with it
    # the node geometry, and the two fields differ at quadrature error
    # (~3e-7 here) even where the hopeless mass itself is 1e-8
    assert tv <= 3.0 * mass + 1e-6
    assert mass < 0.01


@pytest.mark.parametrize("n_hopeless", [50, 500])
def test_adding_hopeless_runners_bounded_impact_blocks(n_hopeless):
    from winning.factor.blocks import block_race_probabilities
    rng = np.random.default_rng(22)
    n_live = 20
    mu = rng.normal(0, 0.8, n_live); mu -= mu.mean()
    D = (0.5 + 0.5 * rng.random(n_live)) ** 2
    v = 0.7 * np.sqrt(D)
    cl = rng.integers(0, 5, n_live)
    p_before = block_race_probabilities(mu, cl, v, D, points=513)
    mu2 = np.r_[mu, 4.0 + 4.0 * rng.random(n_hopeless)]
    D2 = np.r_[D, np.ones(n_hopeless)]
    v2 = np.r_[v, 0.3 * np.ones(n_hopeless)]
    cl2 = np.r_[cl, rng.integers(5, 5 + n_hopeless // 10, n_hopeless)]
    p_after = block_race_probabilities(mu2, cl2, v2, D2, points=513)
    mass = p_after[n_live:].sum()
    live = p_after[:n_live] / p_after[:n_live].sum()
    assert 0.5 * np.abs(live - p_before).sum() <= 3.0 * mass + 1e-9


def test_hopeless_runners_do_not_degrade_lattice_accuracy():
    # the bulk window must keep the closed form accurate at 65 points even
    # with 300 hopeless runners stretching the ability span
    rng = np.random.default_rng(23)
    mu = np.array([0.4, -0.4]); D = np.array([0.6, 1.1]); v = np.array([0.8, 0.3])
    nh = 300
    mu2 = np.r_[mu, 5.0 + 5.0 * rng.random(nh)]
    D2 = np.r_[D, np.ones(nh)]
    v2 = np.r_[v, np.zeros(nh)]
    p = race_probabilities(mu2, V=v2[:, None], D=D2, points=65)
    exact_ratio = norm.cdf((mu[1] - mu[0]) / np.sqrt(D.sum() + (v[0] - v[1]) ** 2))
    pair = p[0] / (p[0] + p[1])
    assert abs(pair - exact_ratio) < 5e-4
