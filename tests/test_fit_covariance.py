"""The dense-covariance front door: race_probabilities(mu, cov=C).

Goldens here are measured, not guessed: the MC referees below were run at
4e5 draws (TV noise ~1e-3 at n=40) when the thresholds were set.
"""
import numpy as np
import pytest

from winning import race_probabilities
from winning.factor.core import fit_covariance, qmc_nodes


def _mc_tv(mu, C, p, draws=400_000, seed=123):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C + 1e-10 * np.eye(len(C)))
    counts = np.bincount(
        (mu + rng.standard_normal((draws, len(C))) @ L.T).argmin(1),
        minlength=len(C)) / draws
    return 0.5 * np.abs(counts - p).sum()


def test_in_grammar_truth_recovered():
    # C already factor+diagonal: the fit must not distort it. The old
    # pipeline (blocks chasing the raw residual's choice-irrelevant common
    # component) scored TV 0.017 here; the projected-residual pipeline
    # measures 1.9e-4, which is lattice/node noise.
    rng = np.random.default_rng(0)
    n = 40
    Vt = rng.normal(size=(n, 2)) * 0.6
    Dt = 0.5 + rng.random(n)
    C = Vt @ Vt.T + np.diag(Dt)
    mu = np.sort(rng.normal(size=n))
    F, W = qmc_nodes(2, 11)
    p_true = race_probabilities(mu, V=Vt, D=Dt, F=F, W=W, points=257)
    p_fit = race_probabilities(mu, cov=C, points=257)
    assert 0.5 * np.abs(p_true - p_fit).sum() < 1e-3


def test_common_component_is_ignored():
    # adding tau^2 11' to Sigma cannot move any win probability; the fit
    # must agree with itself across tau (the identification fact the
    # projected objective encodes).
    rng = np.random.default_rng(1)
    n = 30
    Vt = rng.normal(size=(n, 2)) * 0.5
    C = Vt @ Vt.T + np.diag(0.5 + rng.random(n))
    mu = np.sort(rng.normal(size=n))
    p0 = race_probabilities(mu, cov=C, points=257)
    p9 = race_probabilities(mu, cov=C + 9.0 * np.ones((n, n)), points=257)
    assert 0.5 * np.abs(p0 - p9).sum() < 2e-3


def test_equicorrelation_matches_mc():
    rng = np.random.default_rng(2)
    n = 40
    C = np.full((n, n), 0.6)
    np.fill_diagonal(C, 1.0)
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, cov=C)
    assert abs(p.sum() - 1.0) < 1e-9
    assert _mc_tv(mu, C, p) < 5e-3   # measured 1.1e-3


def test_ar1_within_measured_band():
    # AR(1) is the family the grammar serves worst (locality); measured
    # TV 0.049 at rho=0.9, n=40. This is a regression rail, not a claim
    # of exactness.
    rng = np.random.default_rng(0)
    n = 40
    C = 0.9 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, cov=C)
    assert _mc_tv(mu, C, p) < 0.08


def test_cov_excludes_other_specs():
    with pytest.raises(ValueError):
        race_probabilities([0.0, 1.0], cov=np.eye(2), D=np.ones(2))


def test_grammar_inversion_round_trips():
    # the intro table's claim: priced and inverted by the same two calls,
    # for every grammar member (generic damped inverse, exact forward map)
    from winning.factor.structures import (Blocks, Nested, Tree,
                                           dispatch_probabilities)
    from winning.factor.races import abilities_from_race
    rng = np.random.default_rng(3)
    n = 60
    cluster = np.repeat(np.arange(6), 10)
    mu0 = rng.normal(size=n)
    mu0 -= mu0.mean()
    parent = np.array([6, 6, 7, 7, 8, 8, 8, 8, -1])
    strength = np.concatenate([np.zeros(6), [0.4, 0.4, 0.3]])
    cases = [
        Blocks(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
               D=0.5 + rng.random(n)),
        Nested(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
               D=0.5 + rng.random(n), coupling=0.5 * rng.random(n),
               gamma=0.8),
        Tree(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
             D=0.5 + rng.random(n), parent=parent, strength=strength),
    ]
    for s in cases:
        pstar = dispatch_probabilities(mu0, s)
        mu_hat = abilities_from_race(pstar, structure=s)
        assert np.abs(mu_hat - mu0).max() < 1e-6


def test_cov_inversion_round_trip():
    # invert the fitted race: forward with cov=, invert with cov=,
    # recover the abilities of the fitted model
    rng = np.random.default_rng(4)
    n = 30
    Vt = rng.normal(size=(n, 2)) * 0.5
    C = Vt @ Vt.T + np.diag(0.5 + rng.random(n))
    mu0 = rng.normal(size=n)
    mu0 -= mu0.mean()
    from winning.factor.races import abilities_from_race
    p = race_probabilities(mu0, cov=C)
    mu_hat = abilities_from_race(p, cov=C)
    assert np.abs(mu_hat - mu0).max() < 1e-5
