"""update_market (diagonal prior) must equal the exact full-belief
update_team_market_full(A=I) for heterogeneous market noise (#93)."""

import numpy as np
import pytest

from winning.ratings.market import update_market
from winning.ratings.teams import update_team_market_full

M = np.array([0.3, -0.1, 0.4, -0.6])
V = np.array([0.7, 1.3, 0.4, 2.0])
Y = np.array([1.2, -0.3, 0.5, -1.4])


@pytest.mark.parametrize("tau2", [
    np.full(4, 0.3),
    np.array([0.05, 0.2, 0.8, 1.7]),
    np.array([1e-3, 1.0, 1.0, 10.0]),
])
def test_matches_the_exact_full_update(tau2):
    inv = lambda p: Y
    m, v, lz = update_market(M, V, None, tau2=tau2, invert=inv)
    m2, S2, lz2 = update_team_market_full(M, np.diag(V), np.eye(4), None,
                                          tau2=tau2, invert=inv)
    assert np.allclose(m, m2, atol=1e-12), (m, m2)
    assert np.allclose(v, np.diag(S2), atol=1e-12), (v, np.diag(S2))
    assert abs(lz - lz2) < 1e-12


def test_heterogeneity_moves_the_posterior():
    """The equality must not be two uniform-tau2 updates."""
    inv = lambda p: Y
    a = update_market(M, V, None, tau2=np.array([0.05, 0.2, 0.8, 1.7]),
                      invert=inv)[0]
    b = update_market(M, V, None, tau2=0.6875, invert=inv)[0]
    assert np.abs(a - b).max() > 0.05


def test_a_known_coordinate_with_heterogeneous_tau2():
    """v = 0 is legal (#78); the free block is the conditional given it.
    Referee: a tiny positive variance in the exact update."""
    tau2 = np.array([0.05, 0.2, 0.8, 1.7])
    v0 = V.copy(); v0[1] = 0.0
    inv = lambda p: Y
    m, v, _ = update_market(M, v0, None, tau2=tau2, invert=inv)
    v_eps = v0.copy(); v_eps[1] = 1e-12
    m2, S2, _ = update_team_market_full(M, np.diag(v_eps), np.eye(4), None,
                                        tau2=tau2, invert=inv)
    assert v[1] == 0.0 and m[1] == M[1]
    assert np.allclose(m, m2, atol=1e-8)
    free = [0, 2, 3]
    assert np.allclose(v[free], np.diag(S2)[free], atol=1e-8)
