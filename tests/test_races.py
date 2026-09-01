"""The unified race: special cases recover the named transforms."""

import numpy as np
import pytest

from winning.factor import (abilities_from_race, hermite_nodes,
                            race_probabilities, win_probabilities_factor)
from winning.factor.core import win_probabilities


def _problem(n=40, k=2, seed=7):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n); mu -= mu.mean()
    V = rng.normal(0, 0.4, (n, k))
    D = rng.uniform(0.5, 1.5, n)
    return mu, V, D


def test_zero_factors_is_the_independent_race():
    mu, _, _ = _problem()
    p = race_probabilities(mu)
    q = win_probabilities(mu)
    assert np.abs(p - q).max() < 1e-9


def test_normal_factor_case_is_factor_probit():
    mu, V, D = _problem()
    F, W = hermite_nodes(2)
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    q = win_probabilities_factor(mu, V, D, F, W)
    assert np.abs(p - q).max() < 1e-9


def test_gumbel_zero_loadings_is_softmax():
    # the base is standardized to unit variance, so unit-D Gumbel equals
    # softmax at inverse temperature pi/sqrt(6); D = pi^2/6 gives softmax
    # at temperature one exactly
    mu, _, _ = _problem(n=12)
    c = np.pi / np.sqrt(6.0)
    p = race_probabilities(mu, base="gumbel")
    soft_c = np.exp(-mu * c) / np.exp(-mu * c).sum()
    assert np.abs(p - soft_c).max() < 1e-12
    p1 = race_probabilities(mu, D=np.full(12, np.pi**2 / 6), base="gumbel")
    soft = np.exp(-mu) / np.exp(-mu).sum()
    assert np.abs(p1 - soft).max() < 1e-12


def test_correlated_gumbel_roundtrip():
    mu, V, D = _problem(n=25)
    p = race_probabilities(mu, V=V, D=D, base="gumbel")
    mu_hat = abilities_from_race(p, V=V, D=D, base="gumbel")
    assert np.abs(mu_hat - mu).max() < 1e-4


def test_custom_base_logistic_roundtrip():
    # standardized logistic: variance 1 => scale s = sqrt(3)/pi
    s = np.sqrt(3.0) / np.pi

    def logistic(z):
        u = np.clip(z / s, -700, 700)
        ez = np.exp(-u)
        S = np.maximum(ez / (1 + ez), 1e-300)   # survival of min-wins logistic
        f = np.maximum(ez / (s * (1 + ez) ** 2), 1e-300)
        fp = f * (ez - 1) / (s * (1 + ez))
        return S, f, fp

    mu, _, _ = _problem(n=15)
    p = race_probabilities(mu, base=logistic)
    mu_hat = abilities_from_race(p, base=logistic)
    assert abs(p.sum() - 1) < 1e-12
    assert np.abs(mu_hat - mu).max() < 1e-4


def _mc_softmin(mu, V, D, tau, R=2_000_000, seed=3):
    rng = np.random.default_rng(seed)
    n = len(mu)
    k = V.shape[1] if V is not None else 0
    X = mu[None, :] + np.sqrt(D)[None, :] * rng.normal(0, 1, (R, n))
    if k:
        X += rng.normal(0, 1, (R, k)) @ V.T
    Z = np.exp(-(X - X.min(1, keepdims=True)) / tau)
    return (Z / Z.sum(1, keepdims=True)).mean(0)


def test_temperature_matches_mc_independent():
    mu, _, _ = _problem(n=8)
    D = np.ones(8)
    p = race_probabilities(mu, D=D, temperature=0.7)
    q = _mc_softmin(mu, None, D, 0.7)
    assert abs(p.sum() - 1) < 1e-12
    assert np.abs(p - q).max() < 3e-3          # MC noise ~2e-3


def test_temperature_matches_mc_factor():
    mu, V, D = _problem(n=8)
    F, W = hermite_nodes(2)
    p = race_probabilities(mu, V=V, D=D, F=F, W=W, temperature=0.5)
    q = _mc_softmin(mu, V, D, 0.5)
    assert np.abs(p - q).max() < 3e-3


def test_temperature_limits():
    mu, _, _ = _problem(n=10)
    hard = race_probabilities(mu)
    warm = race_probabilities(mu, temperature=0.05)
    hot = race_probabilities(mu, temperature=25.0)
    assert np.abs(warm - hard).max() < 0.02    # tau -> 0 approaches hard race
    assert np.abs(hot - 0.1).max() < 0.02      # tau -> inf flattens to uniform


def test_temperature_roundtrip():
    mu, V, D = _problem(n=15)
    p = race_probabilities(mu, V=V, D=D, temperature=0.5)
    mu_hat = abilities_from_race(p, V=V, D=D, temperature=0.5)
    assert np.abs(mu_hat - mu).max() < 5e-3


def test_tie_densities_match_committed_vectors():
    import json
    from pathlib import Path
    from winning.factor import tie_densities
    T = json.loads((Path(__file__).parent.parent /
                    "js/factor/test_vectors.json").read_text())
    w = tie_densities(T["problem"]["mu"], V=T["problem"]["V"],
                      D=T["problem"]["D"], F=T["hermite"]["F"],
                      W=T["hermite"]["W"])
    want = np.array(T["expected"]["w"])
    assert np.abs(w - want).max() < 1e-12
    assert np.abs(w - w.T).max() == 0.0


def test_removal_shares_match_core_ensemble():
    from winning.factor import removal_shares
    mu, V, D = _problem(n=20)
    F, W = hermite_nodes(2)
    q = removal_shares(mu, V=V, D=D, F=F, W=W)
    _, q_core = win_probabilities_factor(mu, V, D, F, W,
                                         return_deletions=True)
    assert np.abs(q - q_core).max() < 1e-12
    assert np.abs(q.sum(1) - 1).max() < 1e-12


def test_probit_removal_routes_through_the_safe_path():
    """winning.probit.removal_shares is the max-wins reflection of the
    SAFE removal, not of the raw core deletion rows (sixth review).

    The core rows normalize whatever the field pass produced, with no
    coverage window, no sharpest-runner refinement and no mass check, so
    removing a dominant favourite could return a plausible row the
    lattice never resolved. On a well-conditioned field the two agree to
    machine precision, which is what this pins; the difference is the
    guarantees, not the arithmetic."""
    from winning.probit import removal_shares as probit_removal
    mu, V, D = _problem(n=15)
    q = probit_removal(-mu, V=V, D=D)      # utilities = -abilities
    _, q_core = win_probabilities_factor(mu, V, D, *hermite_nodes(2),
                                         return_deletions=True)
    assert np.abs(q - q_core).max() < 1e-12
    assert np.abs(q.sum(1) - 1).max() < 1e-12

    # and the guarantee the reroute bought: a dominant favourite whose
    # removal moves the winner somewhere the raw rows never covered
    mu2 = mu.copy(); mu2[0] -= 8.0
    q2 = probit_removal(-mu2, V=V, D=D)
    assert np.abs(q2.sum(1) - 1).max() < 1e-9
    assert q2[0, 0] == 0.0 and q2[0].max() > 0.05


def test_top_level_front_door():
    import winning
    mu, V, D = _problem(n=10)
    p = winning.race_probabilities(mu, V=V, D=D)
    mu_hat = winning.calibrate_abilities(p, V=V, D=D)
    assert np.abs(mu_hat - mu).max() < 1e-4
