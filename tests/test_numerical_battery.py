"""The numerical battery: every fast exactness property we can state.

Categories: closed forms, distributional identities, monotonicity and
dominance, structural equivalences (different engines/parametrizations, same
covariance), derivative identities, cross-verb consistency, determinism.
Everything here runs in milliseconds; nothing uses Monte Carlo.
"""
import numpy as np
import pytest
from scipy.stats import norm

from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   block_race_jacobian,
                                   abilities_from_block_race)
from winning.factor.races import (race_probabilities, abilities_from_race,
                                  removal_shares, tie_densities)


def _mk(seed=0, n=14, C=4):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.9, n); mu -= mu.mean()
    D = np.exp(rng.uniform(-0.8, 0.8, n)) ** 2
    v = rng.uniform(0.2, 0.9, n) * np.sqrt(D)
    cl = rng.integers(0, C, n)
    return mu, cl, v, D


# ------------------------------------------------------------ closed forms
def test_gumbel_independent_race_is_exact_softmin():
    # softmin identity needs unit SCALE Gumbel; the base is standardised to
    # unit VARIANCE, so D = pi^2/6 (a convention worth a pinned test in
    # itself: D is variance everywhere, including non-normal bases)
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 1.0, 10); mu -= mu.mean()
    p = race_probabilities(mu, V=None, D=np.full(10, np.pi ** 2 / 6),
                           base="gumbel", points=2001)
    soft = np.exp(-mu) / np.exp(-mu).sum()
    assert 0.5 * np.abs(p - soft).sum() < 2e-3


def test_two_runner_all_structures_closed_form():
    mu = np.array([0.4, -0.4]); D = np.array([0.6, 1.1])
    def exact(var_diff):
        return norm.cdf((mu[1] - mu[0]) / np.sqrt(var_diff))
    # independent
    p = race_probabilities(mu, V=None, D=D, points=2001)
    assert abs(p[0] - exact(D.sum())) < 1e-6
    # factor rank-1: difference variance D1+D2+(v1-v2)^2
    v = np.array([0.8, 0.3])
    p = race_probabilities(mu, V=v[:, None], D=D, points=2001)
    assert abs(p[0] - exact(D.sum() + (v[0] - v[1]) ** 2)) < 1e-5
    # rank-2 blocks, same cluster: ||V1 - V2||^2
    V2 = np.array([[0.5, -0.2], [0.1, 0.4]])
    p = block_race_probabilities(mu, np.array([0, 0]), V2, D, points=1001, qa=11)
    assert abs(p[0] - exact(D.sum() + ((V2[0] - V2[1]) ** 2).sum())) < 1e-4
    # tree, shared ancestor lam: ancestor cancels (common to both)
    parent = np.array([2, 2, -1]); lam = np.array([0.0, 0.0, 0.9])
    p = tree_race_probabilities(mu, np.array([0, 1]), v, D, parent, lam,
                                points=1001)
    assert abs(p[0] - exact(D.sum() + (v ** 2).sum())) < 1e-4


def test_partial_exchangeability():
    # two identical runners + one different: the twins get equal probability
    mu = np.array([0.2, 0.2, -0.4]); D = np.array([0.8, 0.8, 1.0])
    v = np.array([0.5, 0.5, 0.3])
    p = block_race_probabilities(mu, np.array([0, 0, 1]), v, D, points=1001)
    assert abs(p[0] - p[1]) < 1e-10


# -------------------------------------------- monotonicity and dominance
def test_better_ability_more_probability_everywhere():
    mu, cl, v, D = _mk(2)
    p0 = block_race_probabilities(mu, cl, v, D)
    mu2 = mu.copy(); mu2[5] -= 0.1          # min-wins: better
    p1 = block_race_probabilities(mu2, cl, v, D)
    assert p1[5] > p0[5]
    others = np.ones(len(mu), bool); others[5] = False
    assert (p1[others] <= p0[others] + 1e-12).all()


def test_variance_helps_longshots_hurts_favorites():
    n = 8
    mu = np.array([-1.5] + [0.3] * 7)        # runner 0 is a strong favorite
    D = np.ones(n)
    p0 = race_probabilities(mu, V=None, D=D, points=1001)
    D_fav = D.copy(); D_fav[0] = 4.0
    p_fav = race_probabilities(mu, V=None, D=D_fav, points=1001)
    assert p_fav[0] < p0[0]                  # variance hurts the favorite
    mu2 = np.array([1.5] + [-0.2] * 7)       # runner 0 is a longshot
    q0 = race_probabilities(mu2, V=None, D=D, points=1001)
    q_l = race_probabilities(mu2, V=None, D=D_fav, points=1001)
    assert q_l[0] > q0[0]                    # variance helps the longshot


def test_hopeless_runner_does_not_disturb_the_field():
    mu, cl, v, D = _mk(3)
    p0 = block_race_probabilities(mu, cl, v, D)
    mu2 = np.r_[mu, 12.0]                    # min-wins: hopeless
    p1 = block_race_probabilities(mu2, np.r_[cl, cl.max() + 1],
                                  np.r_[v, 0.5], np.r_[D, 1.0])
    assert p1[-1] < 1e-8
    assert 0.5 * np.abs(p1[:-1] / p1[:-1].sum() - p0).sum() < 1e-6


# --------------------------------------- structural equivalences
def test_uniform_coupling_is_invisible():
    mu, cl, v, D = _mk(4)
    p0 = block_race_probabilities(mu, cl, v, D)
    p1 = nested_race_probabilities(mu, cl, v, D,
                                   coupling=np.full(len(mu), 1.3), gamma=1.0,
                                   qf=15)
    assert 0.5 * np.abs(p0 - p1).sum() < 1e-6


def test_rank2_block_with_zero_column_is_rank1():
    mu, cl, v, D = _mk(5)
    V2 = np.column_stack([v, np.zeros(len(v))])
    p1 = block_race_probabilities(mu, cl, v, D)
    p2 = block_race_probabilities(mu, cl, V2, D, qa=9)
    assert 0.5 * np.abs(p1 - p2).sum() < 2e-3


def test_chain_ancestor_folds_into_uniform_loading():
    # one cluster with uniform loading u plus an ancestor of strength lam is
    # the same as loading sqrt(u^2 + lam^2) (both uniform in that cluster)
    n = 10
    rng = np.random.default_rng(6)
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    D = np.ones(n)
    cl = (np.arange(n) >= 5).astype(int)
    u, lam_a = 0.6, 0.8
    v = np.where(cl == 0, u, 0.4)
    parent = np.array([2, 3, 3, -1])         # cluster0 -> node2 -> root3
    lam = np.array([0.0, 0.0, lam_a, 0.0])
    p_tree = tree_race_probabilities(mu, cl, v, D, parent, lam, points=1025)
    v_fold = np.where(cl == 0, np.sqrt(u ** 2 + lam_a ** 2), 0.4)
    p_block = block_race_probabilities(mu, cl, v_fold, D, points=1025)
    assert 0.5 * np.abs(p_tree - p_block).sum() < 1e-3


def test_equal_abilities_variance_wins():
    # at equal abilities the HIGHER-variance runner wins more: the minimum
    # of symmetric variables favors spread. (An earlier draft of this test
    # asserted the opposite ordering and was wrong -- worth pinning.)
    n = 6
    mu = np.zeros(n)
    D = np.array([0.4, 0.6, 0.8, 1.0, 1.4, 2.0])
    p = race_probabilities(mu, V=None, D=D, points=1001)
    assert (np.diff(p) > 0).all()            # monotone in variance
    # and with a common factor the same ordering holds on the idiosyncratic
    v = np.full(n, 0.5)
    p2 = block_race_probabilities(mu, np.zeros(n, int), v, D, points=1001)
    assert (np.diff(p2) > 0).all()


# ------------------------------------------------- derivative identities
def test_jacobian_diagonal_matches_field_slopes():
    rng = np.random.default_rng(8)
    n = 12
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    v = rng.uniform(0.3, 0.8, n); D = np.exp(rng.uniform(-0.5, 0.5, n))
    # C = 1 block with loadings == rank-1 factor race: compare block Jacobian
    # diagonal against the OTHER engine's analytic slopes
    J = block_race_jacobian(mu, np.zeros(n, int), v, D, points=1001)
    _, slopes = race_probabilities(mu, V=v[:, None], D=D, points=1001,
                                   return_slopes=True)
    assert np.corrcoef(np.diag(J), slopes)[0, 1] > 0.999
    assert np.abs(np.diag(J) - slopes).max() < 5e-3


def test_offdiagonal_jacobian_equals_tie_densities():
    # dp_i/dmu_j (i != j) IS the pairwise photo-finish density: the same
    # integral f_i f_j prod_others. Cross-verb identity.
    rng = np.random.default_rng(9)
    n = 10
    mu = rng.normal(0, 0.7, n); mu -= mu.mean()
    v = rng.uniform(0.2, 0.7, n); D = np.exp(rng.uniform(-0.4, 0.4, n))
    J = block_race_jacobian(mu, np.zeros(n, int), v, D, points=1001)
    T = tie_densities(mu, V=v[:, None], D=D, points=1001)
    T = np.asarray(T)
    off = ~np.eye(n, dtype=bool)
    r = J[off] / np.maximum(T[off], 1e-300)
    assert T[off].min() > 0
    assert np.abs(r - np.median(r)).max() < 5e-2 * abs(np.median(r))


def test_removal_shares_match_direct_deletion():
    rng = np.random.default_rng(10)
    n = 9
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    v = rng.uniform(0.2, 0.8, n); D = np.exp(rng.uniform(-0.5, 0.5, n))
    Q = np.asarray(removal_shares(mu, V=v[:, None], D=D, points=1001))
    i = 3
    keep = np.arange(n) != i
    p_direct = race_probabilities(mu[keep], V=v[keep, None], D=D[keep],
                                  points=1001)
    row = Q[i, keep] / Q[i, keep].sum()
    assert 0.5 * np.abs(row - p_direct).sum() < 2e-3


# ------------------------------------------------ inversion cross-checks
def test_two_inverters_agree_on_shared_domain():
    rng = np.random.default_rng(11)
    n = 12
    mu = rng.normal(0, 0.9, n); mu -= mu.mean()
    v = rng.uniform(0.2, 0.8, n); D = np.exp(rng.uniform(-0.5, 0.5, n))
    p = race_probabilities(mu, V=v[:, None], D=D, points=1001)
    mu_a = abilities_from_race(p, V=v[:, None], D=D, points=1001)
    mu_b, resid, _ = abilities_from_block_race(p, np.zeros(n, int), v, D,
                                               points=1001)
    assert np.abs(mu_a - mu_b).max() < 5e-3
    assert resid < 1e-8


# --------------------------------------------------------- determinism
def test_repeated_calls_are_deterministic():
    mu, cl, v, D = _mk(12)
    p1 = block_race_probabilities(mu, cl, v, D)
    p2 = block_race_probabilities(mu, cl, v, D)
    assert np.abs(p1 - p2).max() < 1e-12     # rust rayon reduction included


def test_sharp_factor_race_matches_mc():
    """Fuzz-battery regression (seed 11540): D tiny relative to loadings
    makes the conditional race nearly deterministic; the fixed GH-15 rule
    lost 5e-2 TV. The adaptive order must bring it under 3e-3."""
    from scipy.stats import qmc
    from scipy.special import ndtri
    rng = np.random.default_rng(11540)
    n = 11
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 1)) * 1.2
    D = 0.01 + 0.02 * rng.random(n)
    p = race_probabilities(mu, V=V, D=D, points=1001)
    Sig = V @ V.T + np.diag(D)
    L = np.linalg.cholesky(Sig + 1e-12 * np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1).random_base2(17),
                      1e-12, 1 - 1e-12)).T
    ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                      minlength=n) / z.shape[1]
    assert 0.5 * np.abs(p - ref).sum() < 3e-3
