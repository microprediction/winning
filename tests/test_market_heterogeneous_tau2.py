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


@pytest.mark.parametrize("tau2", [
    np.full(4, 0.3),
    np.array([0.05, 0.2, 0.8, 1.7]),
    np.array([1e-3, 1.0, 1.0, 10.0]),
])
def test_full_covariance_update_matches_too(tau2):
    """#93 follow-up: update_market_full kept P diag(1/tau2) P, 0.23 off
    in the means for heterogeneous tau2, with a correlated prior too."""
    from winning.ratings.full import update_market_full
    inv = lambda p: Y
    rng = np.random.default_rng(93)
    A = rng.normal(size=(4, 4))
    for S0 in (np.diag(V), A @ A.T / 4 + np.diag(V)):
        m, S, lz = update_market_full(M, S0, None, tau2=tau2, invert=inv)
        m2, S2, lz2 = update_team_market_full(M, S0, np.eye(4), None,
                                              tau2=tau2, invert=inv)
        assert np.allclose(m, m2, atol=1e-12)
        assert np.allclose(S, S2, atol=1e-12)
        assert abs(lz - lz2) < 1e-12


@pytest.mark.parametrize("scale", [1e-10, 1e-22])
def test_small_heterogeneous_tau2_takes_the_contrast_branch(scale):
    """#93 after #715: np.allclose(tau2, tau2[0]) is True for
    [1e-12, 9e-9, 2e-12, 8e-9] (atol 1e-8), so both updates used the
    scalar-noise formula: means 0.77 (full) and 0.95 (diagonal) off the
    exact team/identity lift. Only exactly equal tau2 is uniform now.
    logZ (~1e9..1e11 here) agrees to its own roundoff, not compared."""
    from winning.ratings.full import update_market_full
    m0 = np.array([.3, -.1, .4, -.6])
    y = np.array([1.2, -.3, .5, -1.4])
    inv = lambda p: y
    tau2 = np.array([1e-12, 9e-9, 2e-12, 8e-9])
    v0 = np.array([1, 2, .5, 3]) * scale
    mt, St, _ = update_team_market_full(m0, np.diag(v0), np.eye(4), None,
                                        tau2=tau2, invert=inv)
    mf, Sf, _ = update_market_full(m0, np.diag(v0), None, tau2=tau2,
                                   invert=inv)
    md, vd, _ = update_market(m0, v0, None, tau2=tau2, invert=inv)
    assert np.abs(mf - mt).max() < 1e-12 and np.abs(md - mt).max() < 1e-12
    assert np.abs(Sf - St).max() < 1e-9 * scale
    assert np.abs(vd - np.diag(St)).max() < 1e-9 * scale
