"""Tests for winning.factor.polish: the race Jacobian and concentration
polishing (the core transport primitive; allocation etc. are thin users)."""
import numpy as np
import pytest

from winning.factor.polish import (race_jacobian, polish_race,
                                   concentration_matrix)
from winning.factor.races import race_probabilities


def _problem(seed=5, n=12):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    g = rng.normal(0, 0.6, n); g -= g.mean()
    D = 0.5 + 0.5 * rng.random(n)
    return mu, g[:, None], D


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_jacobian_matches_finite_differences(base):
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D, base=base)
    J = race_jacobian(mu, V=V, D=D, base=base)
    for j in range(len(mu)):
        m2 = mu.copy(); m2[j] += 1e-6
        fd = (race_probabilities(m2, V=V, D=D, base=base) - p0) / 1e-6
        assert np.abs(fd - J[:, j]).max() < 5e-6
    assert np.abs(J.sum(axis=1)).max() < 1e-12   # shift invariance


def test_polish_hits_caps_and_stays_a_race():
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D)
    big = int(np.argmax(p0))
    sector = [int(i) for i in np.argsort(-p0)[:4]]
    caps = np.full(len(mu), np.nan); caps[big] = 0.25
    p, mu1, info = polish_race(mu0=mu, V=V, D=D, name_caps=caps,
                               groups=[(sector, 0.60)])
    assert p[big] <= 0.25 + 1e-8
    assert p[sector].sum() <= 0.60 + 1e-8
    assert info["max_violation"] < 1e-8
    # the deliverable property: output IS the race at the returned abilities
    assert np.abs(p - race_probabilities(mu1, V=V, D=D)).max() < 1e-12
    # binding constraints are active, others untouched in direction
    assert len(info["active"]) >= 1


def test_polish_noop_when_caps_slack():
    mu, V, D = _problem()
    p0 = race_probabilities(mu, V=V, D=D)
    p, mu1, info = polish_race(mu0=mu, V=V, D=D, name_caps=np.full(len(mu), 0.99))
    assert np.abs(p - p0).max() < 1e-6
    assert info["mu_distance"] < 1e-4


def test_polish_from_probabilities_entrypoint():
    mu, V, D = _problem(seed=7)
    p0 = race_probabilities(mu, V=V, D=D)
    p, mu1, info = polish_race(p0=p0, V=V, D=D,
                               name_caps=np.where(np.arange(len(mu)) == np.argmax(p0),
                                                  0.8 * p0.max(), np.nan))
    assert p.max() <= 0.8 * p0.max() + 1e-6
    assert abs(p.sum() - 1.0) < 1e-9


def test_concentration_matrix_shapes():
    A, b = concentration_matrix(5, name_caps=[0.3, np.nan, 0.2, np.nan, np.nan],
                                groups=[([0, 1], 0.5)])
    assert A.shape == (3, 5) and len(b) == 3
    assert b[-1] == 0.5 and A[-1, :2].sum() == 2.0
