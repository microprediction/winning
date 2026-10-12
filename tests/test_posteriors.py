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


# ---------------------------------------------------------------- #634

@pytest.mark.parametrize("S", [np.array([[1.0, 2.0], [2.0, 1.0]]),
                               np.array([[1.0, 10.0], [0.0, 1.0]])],
                         ids=["indefinite", "asymmetric"])
def test_update_contrast_refuses_a_matrix_that_is_not_a_covariance_634(S):
    with pytest.raises(ValueError, match="cov"):
        wf.update_contrast(np.zeros(2), S, np.array([1.0, -1.0]),
                           mean=0.0, var=1.0)


def test_update_contrast_accepts_a_singular_covariance_634():
    # a degenerate but valid prior (both coordinates equal) is conditioning
    m, S2 = wf.update_contrast(np.zeros(2), np.ones((2, 2)),
                               np.array([1.0, 0.0]), mean=1.0, var=1.0)
    np.testing.assert_allclose(m, [0.5, 0.5])
    np.testing.assert_allclose(S2, np.full((2, 2), 0.5))


# ---------------------------------------------------------------- #669

def _exact_contrast_posterior(V, D, a, var):
    """Sigma - g g'/s in exact rational arithmetic."""
    from fractions import Fraction as Fr
    K, rho = len(D), (len(V[0]) if len(V) and len(V[0]) else 0)
    S = [[sum(Fr(V[i][k]) * Fr(V[j][k]) for k in range(rho))
          + (Fr(D[i]) if i == j else 0) for j in range(K)] for i in range(K)]
    g = [sum(S[i][j] * Fr(a[j]) for j in range(K)) for i in range(K)]
    s = sum(Fr(a[i]) * g[i] for i in range(K)) + Fr(var)
    return [[float(S[i][j] - g[i] * g[j] / s) for j in range(K)]
            for i in range(K)], float(Fr(var) * (s - Fr(var)) / s)


@pytest.mark.parametrize("a", [[1, 0, 0, 0], [1, -1, 0, 0], [1, -1, 2, 0],
                               [0.5, 2, -1, 3]])
@pytest.mark.parametrize("rho", [0, 2])
@pytest.mark.parametrize("var", [1e-3, 1e-12, 1e-16, 1e-20, 1e-30])
def test_update_contrast_stays_finite_and_exact_for_a_precise_observation_669(a, rho, var):
    # var/(a'Da) at and far below machine epsilon: the residual on a's
    # support was formed by subtraction, rounded to exactly zero, and a
    # 0/0 put NaN in D'. Checked against exact rational arithmetic.
    Vl = [[0.5, -0.25], [0.25, 0.5], [-0.5, 0.125], [0.125, 0.25]]
    Vm = np.array(Vl)[:, :rho] if rho else np.zeros((4, 0))
    Dl = [1.0, 0.5, 2.0, 0.75]
    av = np.array(a, dtype=float)
    m, V1, D1, F1, W1, rep = wf.update_contrast(
        np.zeros(4), (Vm, np.array(Dl)), av, mean=1.0, var=var)
    assert rep["exact"]
    assert np.isfinite(V1).all() and np.isfinite(D1).all() and (D1 > 0).all()
    got = V1 @ V1.T + np.diag(D1)
    want, want_aSa = _exact_contrast_posterior(
        Vm.tolist() if rho else [[] for _ in range(4)], Dl, a, var)
    np.testing.assert_allclose(got, np.array(want), rtol=0, atol=1e-12)
    # The observed direction, read without cancellation (a sum of
    # squares). A covariance held as order-one doubles cannot carry a
    # variance of 1e-30 in a mixed direction, so the claim is absolute.
    direct = float(np.sum((V1.T @ av) ** 2) + np.sum(av ** 2 * D1))
    assert abs(direct - want_aSa) < 1e-13
    if np.count_nonzero(av) == 1:
        # one entrant observed: its variance lives in D' alone and is
        # exact to rounding at every var, D var / (D a^2 + var)
        i = int(np.flatnonzero(av)[0])
        np.testing.assert_allclose(
            D1[i], Dl[i] * var / (Dl[i] * av[i] ** 2 + var), rtol=1e-12)
    p = wf.race_probabilities(-m, V=-V1, D=D1)        # consumable downstream
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-9


# ---------------------------------------------------------------- #670

def test_representable_posterior_at_the_bottom_of_the_float_range_670():
    # every input is a normal float and the exact posterior variance,
    # c^2 / 11 = 3.6e-309, is representable; n/D overflowed and both
    # constructors returned d = 0 (the bandit also lost its mean)
    c = 2e-154
    D = np.full(2, c * c)
    m, d, _ = crn_posterior(np.zeros((2, 0)), D, n=10,
                            ybar=np.array([c, -c]), s0=c)
    np.testing.assert_allclose(m / c, [10 / 11, -10 / 11], rtol=1e-14)
    np.testing.assert_allclose(d, c * c / 11, rtol=1e-12)
    m, d, _ = feature_bandit_posterior(np.zeros((2, 0)), D,
                                       counts=np.array([10.0, 0.0]),
                                       sums=np.array([10 * c, 0.0]),
                                       sigma_n=c)
    np.testing.assert_allclose(m / c, [10 / 11, 0.0], rtol=1e-14, atol=0)
    np.testing.assert_allclose(d, [c * c / 11, c * c], rtol=1e-12)


@pytest.mark.parametrize("c", [2e-154, 1e-100, 1e-20, 1e20, 1e100, 1e150])
def test_all_three_constructors_are_scale_equivariant_670(c):
    # locations scale by c and variances by c^2; the answer must follow
    # exactly, at the ends of the float range as in the middle
    rng = np.random.default_rng(3)
    K, rho = 5, 2
    V = rng.normal(size=(K, rho))
    D = rng.uniform(0.3, 1.5, K)
    s0 = rng.uniform(0.6, 1.4, K)
    yb = rng.normal(size=K)
    Y = rng.normal(size=(K, 9))
    Y[rng.random((K, 9)) < 0.3] = np.nan
    cnt = np.array([4.0, 0.0, 7.0, 1.0, 3.0])
    sm = np.where(cnt > 0, rng.normal(size=K) * cnt, 0.0)
    sn = rng.uniform(0.5, 1.5, K)
    cases = [
        (crn_posterior, (V, D, 6, yb, s0), (V * c, D * c * c, 6, yb * c, s0 * c)),
        (crn_posterior_replicates, (V, D, Y, s0), (V * c, D * c * c, Y * c, s0 * c)),
        (feature_bandit_posterior, (V, D, cnt, sm, sn),
         (V * c, D * c * c, cnt, sm * c, sn * c)),
    ]
    for fn, unit_args, scaled_args in cases:
        m1, d1, W1 = fn(*unit_args)
        m2, d2, W2 = fn(*scaled_args)
        np.testing.assert_allclose(m2 / c, m1, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(d2 / c / c, d1, rtol=1e-12)
        np.testing.assert_allclose((W2 / c) @ (W2 / c).T, W1 @ W1.T,
                                   rtol=1e-12, atol=1e-14)
