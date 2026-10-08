"""crn_posterior, crn_posterior_replicates, feature_bandit_posterior (#624):
the exact factor-form posteriors of papers/exact_pom Section 3, checked
against the research scripts they were ported from and against dense
Gaussian conditioning written out independently."""
import importlib.util
import pathlib

import numpy as np
import pytest

import winning
import winning.factor as wf
from winning.factor import (crn_posterior, crn_posterior_replicates,
                            feature_bandit_posterior)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(rel):
    spec = importlib.util.spec_from_file_location(
        rel.replace("/", "_").replace(".py", ""), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dense_crn(V, D, Y, s0, m0):
    """Stack every observed Y_ir; condition theta ~ N(m0, diag(s0^2))
    on it with the full replicate covariance (block per replicate)."""
    K = len(m0)
    P = np.diag(np.broadcast_to(s0, (K,)) ** 2)
    rows = [np.flatnonzero(~np.isnan(Y[:, r])) for r in range(Y.shape[1])]
    N = sum(len(S) for S in rows)
    A, C, y, p = np.zeros((N, K)), np.zeros((N, N)), np.zeros(N), 0
    for r, S in enumerate(rows):
        q = p + len(S)
        A[np.arange(p, q), S] = 1.0
        C[p:q, p:q] = V[S] @ V[S].T + np.diag(D[S])
        y[p:q] = Y[S, r]
        p = q
    G = P @ A.T @ np.linalg.inv(A @ P @ A.T + C)
    return m0 + G @ (y - A @ m0), P - G @ A @ P


def _dense_bandit(V0, d0, counts, sums, sn):
    S0 = V0 @ V0.T + np.diag(d0)
    prec = np.linalg.inv(S0) + np.diag(counts / sn ** 2)
    S = np.linalg.inv(prec)
    return S @ (sums / sn ** 2), S


@pytest.fixture
def setup():
    rng = np.random.default_rng(624)
    K, rho = 9, 2
    return (rng, rng.normal(size=(K, rho)), rng.uniform(0.3, 1.2, K),
            rng.uniform(0.5, 2.0, K), rng.normal(size=K))


def test_exported():
    for f in ("crn_posterior", "crn_posterior_replicates",
              "feature_bandit_posterior"):
        assert getattr(wf, f) is getattr(winning, f)


# ---------------------------------------------------------------- research
@pytest.mark.parametrize("kind,c", [("aligned", 0.6), ("opposed", 0.6),
                                    ("independent", 0.0)])
def test_crn_reproduces_the_research_script(kind, c):
    rs = _load("research/rs_crn/exp1_stopping/run_stopping.py")
    V, sigma = rs.make_config(kind, c)
    rng = np.random.default_rng(1)
    for n in (1, 7, 60):
        ybar = rng.normal(size=rs.K)
        mu_r, W_r, d_r, sd_r = rs.posterior_factor_form(n, ybar, V, sigma)
        m, d, W = crn_posterior(V, sigma ** 2, n, ybar, rs.S0)
        assert W.shape == W_r.shape
        np.testing.assert_allclose(d, d_r, rtol=1e-13)
        np.testing.assert_allclose(m, mu_r, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(W @ W.T, W_r @ W_r.T, atol=1e-13)
        np.testing.assert_allclose(np.sqrt(d + (W ** 2).sum(1)), sd_r,
                                   rtol=1e-12)


@pytest.mark.parametrize("kind,c", [("clusters", 0.7), ("none", 0.0)])
def test_bandit_reproduces_the_research_script(kind, c):
    rb = _load("research/rs_crn/exp2_bandit/run_bandit.py")
    V, d0 = rb.make_prior(kind, c)
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 6, rb.N).astype(float)
    sums = np.where(counts > 0, rng.normal(size=rb.N) * counts, 0.0)
    m_r, W_r, d_r = rb.posterior(V, d0, counts, sums)
    m, d, W = feature_bandit_posterior(V, d0, counts, sums, rb.SIGMA_N)
    np.testing.assert_allclose(d, d_r, rtol=1e-13)
    np.testing.assert_allclose(m, m_r, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(W @ W.T, W_r @ W_r.T, atol=1e-13)


# ------------------------------------------------------------ dense truth
@pytest.mark.parametrize("n", [1, 3, 20])
def test_crn_equals_dense_conditioning(setup, n):
    rng, V, D, s0, m0 = setup
    Y = rng.normal(size=(len(D), n)) + 3.0
    m, d, W = crn_posterior(V, D, n, Y.mean(axis=1), s0, m0)
    mr, Sr = _dense_crn(V, D, Y, s0, m0)
    assert W.shape == V.shape
    np.testing.assert_allclose(m, mr, atol=1e-12)
    np.testing.assert_allclose(np.diag(d) + W @ W.T, Sr, atol=1e-12)


def test_unequal_counts_exact_via_replicates(setup):
    rng, V, D, s0, m0 = setup
    K = len(D)
    Y = rng.normal(size=(K, 6))
    Y[0, 2:] = np.nan          # nested: entrant 0 ran only 2
    Y[3, 4:] = np.nan
    Y[5, [1, 3]] = np.nan      # and an arbitrary missing pattern
    Y[7, :] = np.nan           # never run: keeps its prior marginal
    m, d, W = crn_posterior_replicates(V, D, Y, s0, m0)
    mr, Sr = _dense_crn(V, D, Y, s0, m0)
    np.testing.assert_allclose(m, mr, atol=1e-12)
    np.testing.assert_allclose(np.diag(d) + W @ W.T, Sr, atol=1e-12)
    assert d[7] == pytest.approx(s0[7] ** 2)


def test_unequal_means_are_not_sufficient(setup):
    """The reason crn_posterior refuses unequal n_i: two data sets with
    the SAME counts and per-entrant means give different posteriors."""
    rng, V, D, s0, m0 = setup
    Y1 = rng.normal(size=(len(D), 2))
    Y1[1, 1] = np.nan
    Y2 = Y1.copy()
    Y2[0, 0], Y2[0, 1] = Y1[0, 1], Y1[0, 0]   # swap entrant 0's replicates
    a = crn_posterior_replicates(V, D, Y1, s0, m0)[0]
    b = crn_posterior_replicates(V, D, Y2, s0, m0)[0]
    assert np.abs(a - b).max() > 1e-3
    with pytest.raises(ValueError, match="crn_posterior_replicates"):
        crn_posterior(V, D, np.r_[2, 1, np.full(len(D) - 2, 2)],
                      np.nanmean(Y1, axis=1), s0, m0)


def test_replicates_with_full_data_is_crn_posterior(setup):
    rng, V, D, s0, m0 = setup
    Y = rng.normal(size=(len(D), 5))
    m1, d1, W1 = crn_posterior(V, D, np.full(len(D), 5), Y.mean(1), s0, m0)
    m2, d2, W2 = crn_posterior_replicates(V, D, Y, s0, m0)
    np.testing.assert_allclose(m1, m2, atol=1e-13)
    np.testing.assert_allclose(d1, d2, rtol=1e-14)
    np.testing.assert_allclose(W1 @ W1.T, W2 @ W2.T, atol=1e-13)


def test_bandit_equals_dense_conditioning(setup):
    rng, V, D, _, _ = setup
    counts = np.array([0, 1, 4, 0, 2, 9, 1, 3, 0], float)
    sums = np.where(counts > 0, rng.normal(size=9) * 2, 0.0)
    sn = rng.uniform(0.5, 1.5, 9)
    m, d, W = feature_bandit_posterior(V, D, counts, sums, sn)
    mr, Sr = _dense_bandit(V, D, counts, sums, sn)
    np.testing.assert_allclose(m, mr, atol=1e-12)
    np.testing.assert_allclose(np.diag(d) + W @ W.T, Sr, atol=1e-12)


def test_bandit_is_sequential_crn_like(setup):
    """Feature bandit with every arm pulled once under prior factors is
    the same algebra as one CRN replicate with no shared factor: a
    sanity link between the two forms (independent prior, V = 0)."""
    rng, _, D, s0, m0 = setup
    K = len(D)
    y = rng.normal(size=K)
    m1, d1, _ = crn_posterior(np.zeros((K, 1)), D, 1, y, s0)
    m2, d2, _ = feature_bandit_posterior(np.zeros((K, 1)), s0 ** 2,
                                         np.ones(K), y, np.sqrt(D))
    np.testing.assert_allclose(m1, m2, atol=1e-14)
    np.testing.assert_allclose(d1, d2, rtol=1e-14)


# ------------------------------------------------------------- invariances
def test_rotation_and_permutation(setup):
    rng, V, D, s0, m0 = setup
    K = len(D)
    Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    ybar = rng.normal(size=K)
    m, d, W = crn_posterior(V, D, 4, ybar, s0, m0)
    mq, dq, Wq = crn_posterior(V @ Q, D, 4, ybar, s0, m0)
    np.testing.assert_allclose(Wq, W @ Q, atol=1e-13)
    np.testing.assert_allclose(mq, m, atol=1e-13)
    p = rng.permutation(K)
    mp, dp, Wp = crn_posterior(V[p], D[p], 4, ybar[p], s0[p], m0[p])
    np.testing.assert_allclose(mp, m[p], atol=1e-13)
    np.testing.assert_allclose(Wp, W[p], atol=1e-13)
    counts = rng.integers(0, 4, K).astype(float)
    sums = counts * rng.normal(size=K)
    mb, db, Wb = feature_bandit_posterior(V, D, counts, sums, 1.0)
    mbq, _, Wbq = feature_bandit_posterior(V @ Q, D, counts, sums, 1.0)
    np.testing.assert_allclose(Wbq, Wb @ Q, atol=1e-13)
    np.testing.assert_allclose(mbq, mb, atol=1e-13)
    mbp, dbp, Wbp = feature_bandit_posterior(V[p], D[p], counts[p],
                                             sums[p], 1.0)
    np.testing.assert_allclose(mbp, mb[p], atol=1e-13)
    np.testing.assert_allclose(Wbp, Wb[p], atol=1e-13)


def test_zero_replicates_is_the_prior(setup):
    _, V, D, s0, m0 = setup
    m, d, W = crn_posterior(V, D, 0, np.zeros(len(D)), s0, m0)
    np.testing.assert_array_equal(m, m0)
    np.testing.assert_allclose(d, s0 ** 2)
    assert not W.any()


def test_posterior_prices_as_a_race(setup):
    """The output plugs straight into the engine (max race by negation)."""
    rng, V, D, s0, m0 = setup
    m, d, W = crn_posterior(V, D, 3, rng.normal(size=len(D)), s0, m0)
    p = winning.race_probabilities(-m, V=-W, D=d)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------- validation
def test_validation(setup):
    _, V, D, s0, m0 = setup
    K = len(D)
    y = np.zeros(K)
    with pytest.raises(ValueError):
        crn_posterior(V, np.r_[0.0, D[1:]], 2, y, s0)     # Dirac noise
    with pytest.raises(ValueError):
        crn_posterior(V, D, 2, y, 0.0)                    # Dirac prior
    with pytest.raises(ValueError):
        crn_posterior(V, D, -1, y, s0)
    with pytest.raises(ValueError):
        crn_posterior(V, D, 1.5, y, s0)
    with pytest.raises(ValueError):
        crn_posterior(V, D, 2, np.r_[np.nan, y[1:]], s0)
    with pytest.raises(ValueError):
        crn_posterior(V, D, 2, y, s0, m0=np.zeros(K + 1))
    with pytest.raises(ValueError):
        crn_posterior_replicates(V, D, np.full((K, 2), np.inf), s0)
    with pytest.raises(ValueError):
        feature_bandit_posterior(V, D, -np.ones(K), y, 1.0)
    with pytest.raises(ValueError):
        feature_bandit_posterior(V, D, np.zeros(K), np.ones(K), 1.0)
    with pytest.raises(ValueError):
        feature_bandit_posterior(V, D, np.ones(K), y, 0.0)
    with pytest.raises(ValueError):
        feature_bandit_posterior(V, np.zeros(K), np.ones(K), y, 1.0)
