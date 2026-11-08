"""A Gaussian network with win nodes on arbitrary subsets and linear maps
of its variables: every message updates the whole belief."""
import numpy as np
import pytest

from winning.ratings.net import GaussianNet, random_walk_cov
from winning.ratings.full import update_winner_full
from winning.ratings.nway import update_topset_correlated
from winning.factor.races import race_probabilities


def _net5():
    rng = np.random.default_rng(0)
    Bm = rng.normal(size=(5, 2)); S = Bm @ Bm.T * 0.3 + np.eye(5) * 0.8
    m = rng.normal(size=5) * 0.3
    return GaussianNet(list("abcde"), m, S), m, S


def test_a_node_on_the_whole_field_is_the_dense_update():
    net, m, S = _net5()
    net.attach("race", members=list("abcde"), beta2=1.0)
    p = net.query("race")
    # two quadrature routes to one orthant agree to their node rules' accuracy
    np.testing.assert_allclose(p, race_probabilities(-m, cov=S + np.eye(5)), atol=1e-3)
    info = net.observe("race", winner="d")
    m_ref, S_ref, lz = update_winner_full(m, S, 3, beta2=1.0)
    # the forward probability of the winner IS the evidence of observing it.
    # On a correlated belief both sides split the covariance numerically and
    # quadrature it with different rules (Gauss-Hermite 7 against 9, 257
    # against 801 lattice points), so they agree to the rules' accuracy:
    # 2e-5 on this machine, 1.05e-4 on CI's Ubuntu 3.13, where the rank fit
    # lands differently. On a diagonal belief neither side quadratures and
    # the identity is tight.
    assert abs(p[3] - np.exp(lz)) < 5e-4, (p[3], np.exp(lz))
    netd = GaussianNet(list("abcde"), m, np.diag(np.diag(S)))
    netd.attach("race", members=list("abcde"), beta2=1.0)
    pd = netd.query("race")
    lzd = netd.observe("race", winner="d")["logZ_outcome"]
    assert abs(pd[3] - np.exp(lzd)) < 1e-6, (pd[3], np.exp(lzd))
    np.testing.assert_allclose(net.m, m_ref, atol=1e-10); np.testing.assert_allclose(net.S, S_ref, atol=1e-10)
    assert info["logZ_outcome"] == lz and net.evidence == lz


def test_evidence_on_a_subset_moves_the_correlated_outsiders():
    rng = np.random.default_rng(1)
    # two variables outside the node correlated with members, two not
    S = np.eye(6); S[0, 4] = S[4, 0] = 0.7; S[1, 5] = S[5, 1] = -0.6
    net = GaussianNet(list("abcdef"), np.zeros(6), S)
    net.attach("heat", members=list("abcd"), beta2=0.5)
    net.observe("heat", top=["a", "b"])
    assert net.m[4] > 0.1 and net.m[5] < -0.1        # e moves with a, f against b
    # Monte Carlo referee on the joint: rejection on the top set
    s = rng.multivariate_normal(np.zeros(6), S, size=3_000_000)
    X = s[:, :4] + np.sqrt(0.5) * rng.normal(size=(len(s), 4))
    keep = np.array([set(r[:2]) == {0, 1} for r in np.argsort(-X, axis=1)])
    m_mc = s[keep].mean(axis=0); se = s[keep].std(axis=0) / np.sqrt(keep.sum())
    assert np.abs(net.m - m_mc).max() < 4 * se.max() + 4e-3, (net.m, m_mc)
    assert np.abs(np.diag(net.S) - s[keep].var(axis=0)).max() < 0.02


def test_multitemporal_a_win_at_one_date_moves_the_other_dates():
    # three players, four dates each, random-walk prior per player
    T = 4; names = [f"{p}{t}" for p in "xyz" for t in range(T)]
    S = np.kron(np.eye(3), random_walk_cov(T, init_var=1.0, drift=0.2))
    net = GaussianNet(names, np.zeros(12), S)
    net.attach("date2", members=["x2", "y2", "z2"], beta2=1.0)
    net.observe("date2", winner="x2")
    xs = [net.m[net.index[f"x{t}"]] for t in range(T)]
    assert all(v > 0 for v in xs) and xs[2] == max(xs)       # the win lifts x at every date, most at its own
    assert net.m[net.index["x0"]] > net.m[net.index["x3"]] * 0.5
    # a later node reads the smoothed belief
    net.attach("date3", members=["x3", "y3", "z3"], beta2=1.0)
    p3 = net.query("date3")
    assert p3[0] > 1 / 3


def test_sum_nodes_prices_and_dnf():
    # a four-round total as one node: A sums each player's rounds
    names = [f"{p}{r}" for p in "pqr" for r in range(4)]
    S = np.kron(np.eye(3), np.eye(4) * 0.5)
    net = GaussianNet(names, np.zeros(12), S)
    A = np.kron(np.eye(3), np.ones((1, 4)))
    net.attach("total", A=A, beta2=4.0)
    p0 = net.query("total")
    np.testing.assert_allclose(p0, [1 / 3] * 3, atol=1e-6)
    net.observe("total", prices=[0.5, 0.3, 0.2], tau2=0.2)
    assert net.m[0] > 0 and net.m[8] < 0
    p1 = net.query("total")
    assert p1[0] > p1[1] > p1[2]
    # a withdrawal: the outcome runs on the other two, the third is untouched by it
    m_before = net.m.copy()
    net.observe("total", winner=1, dnf=[2])
    assert np.allclose(net.m[8:], m_before[8:]) and net.m[4] > m_before[4]
    with pytest.raises(ValueError, match="not in the field"):
        net.observe("total", winner=2, dnf=[2])


def test_subset_topset_matches_the_diagonal_member_on_a_diagonal_belief():
    v = np.array([0.8, 1.0, 0.6, 1.2, 0.9]); m = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
    net = GaussianNet(list("abcde"), m, np.diag(v))
    V = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])
    net.attach("race", members=list("abcde"), beta2=1.0, V=V)
    net.observe("race", top=["d", "a"])
    m_ref, v_ref, lz = update_topset_correlated(m, v, [3, 0], V=V, beta2=1.0)
    np.testing.assert_allclose(net.m, m_ref, atol=3e-3)
    np.testing.assert_allclose(np.diag(net.S), v_ref, rtol=3e-2)
