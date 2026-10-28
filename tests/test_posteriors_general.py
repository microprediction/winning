"""crn_posterior_replicates' keyword extensions (#635, #636, #638): a
factor or dense prior, per-cell noise and replicate counts, shared shock
levels (trial and seed) and a free column level. Every one is claimed
EXACT, so every one is checked against dense Gaussian conditioning on the
stacked observations, written out independently of the factor algebra."""
import numpy as np
import pytest

from winning.factor import crn_posterior_replicates, feature_bandit_posterior


def _dense(cells, S0, m0, shocks, free=False):
    """Condition theta ~ N(m0, S0) on cells = [(i, r, y, var)], where
    Cov(cell j, cell k) = sum over shocks (labels, U) of
    [labels[r_j] == labels[r_k]] U_i . U_k, plus var on the diagonal.
    free=True: each column also has a flat-prior level, so condition on
    within-column contrasts only (their likelihood is the flat-prior
    marginal likelihood exactly)."""
    K = len(m0)
    N = len(cells)
    ii = np.array([c[0] for c in cells])
    rr = np.array([c[1] for c in cells])
    y = np.array([c[2] for c in cells], dtype=float)
    A = np.zeros((N, K))
    A[np.arange(N), ii] = 1.0
    C = np.diag([c[3] for c in cells]).astype(float)
    for lab, U in shocks:
        lab = np.asarray(lab)
        same = lab[rr][:, None] == lab[rr][None, :]
        C += same * (U[ii] @ U[ii].T)
    if free:
        rows = []
        for r in np.unique(rr):
            js = np.flatnonzero(rr == r)
            for j in js[1:]:
                e = np.zeros(N)
                e[j], e[js[0]] = 1.0, -1.0
                rows.append(e)
        M = np.array(rows).reshape(-1, N)
        y, A, C = M @ y, M @ A, M @ C @ M.T
    G = S0 @ A.T @ np.linalg.inv(A @ S0 @ A.T + C)
    return m0 + G @ (y - A @ m0), S0 - G @ A @ S0


def _cells(Y, Dc):
    K, R = Y.shape
    return [(i, r, Y[i, r], Dc[i, r]) for r in range(R) for i in range(K)
            if not np.isnan(Y[i, r])]


def _close(post, ref, tol=1e-11):
    m, d, W = post
    mr, Sr = ref
    scale = max(1.0, float(np.abs(Sr).max()))
    np.testing.assert_allclose(m, mr, atol=tol * max(1.0, np.abs(mr).max()))
    np.testing.assert_allclose(np.diag(d) + W @ W.T, Sr, atol=tol * scale)
    assert (d > 0).all()


@pytest.fixture
def world():
    rng = np.random.default_rng(635)
    K, rho, R = 8, 2, 7
    V = rng.normal(size=(K, rho))
    D = rng.uniform(0.3, 1.2, K)
    Y = rng.normal(size=(K, R)) + 2.0
    Y[0, 3:] = np.nan
    Y[2, [1, 4]] = np.nan
    Y[5, :2] = np.nan
    Y[6, 6] = np.nan
    return rng, K, rho, R, V, D, Y


# ------------------------------------------------------------ old path
def test_general_path_reproduces_the_original(world):
    """V=None-free calls with an independent prior route through the
    original code; the general engine, forced by an equivalent keyword
    (D as a matrix), must agree with it."""
    rng, K, rho, R, V, D, Y = world
    s0 = rng.uniform(0.5, 2.0, K)
    m0 = rng.normal(size=K)
    old = crn_posterior_replicates(V, D, Y, s0, m0)
    new = crn_posterior_replicates(V, np.tile(D[:, None], (1, R)), Y, s0, m0)
    np.testing.assert_allclose(new[0], old[0], atol=1e-12)
    np.testing.assert_allclose(new[1], old[1], rtol=1e-14)
    np.testing.assert_allclose(new[2] @ new[2].T, old[2] @ old[2].T,
                               atol=1e-12)


# --------------------------------------------------------------- #635
@pytest.mark.parametrize("q", [1, 3])
def test_factor_prior_is_exact_635(world, q):
    rng, K, rho, R, V, D, Y = world
    V0 = rng.normal(size=(K, q))
    d0 = rng.uniform(0.2, 1.5, K)
    m0 = rng.normal(size=K)
    post = crn_posterior_replicates(V, D, Y, prior=(V0, d0, m0))
    S0 = V0 @ V0.T + np.diag(d0)
    ref = _dense(_cells(Y, np.tile(D[:, None], (1, R))), S0, m0,
                 [(np.arange(R), V)])
    _close(post, ref)
    # d is the e-part of the posterior: 1/(1/d0 + n_i/D_i), as claimed
    n = (~np.isnan(Y)).sum(1)
    np.testing.assert_allclose(post[1], 1 / (1 / d0 + n / D), rtol=1e-13)
    # m0 may come by keyword instead of inside the tuple
    post2 = crn_posterior_replicates(V, D, Y, m0=m0, prior=(V0, d0))
    np.testing.assert_array_equal(post2[0], post[0])


def test_dense_prior_is_exact_635(world):
    rng, K, rho, R, V, D, Y = world
    A = rng.normal(size=(K, K))
    S0 = A @ A.T / K + 0.1 * np.eye(K)
    m0 = rng.normal(size=K)
    post = crn_posterior_replicates(V, D, Y, m0=m0, prior=S0)
    # a nested list is the dense matrix too, never a (V0, d0) pair
    post_l = crn_posterior_replicates(V, D, Y, m0=m0, prior=S0.tolist())
    np.testing.assert_array_equal(post_l[0], post[0])
    ref = _dense(_cells(Y, np.tile(D[:, None], (1, R))), S0, m0,
                 [(np.arange(R), V)])
    _close(post, ref)
    assert post[2].shape[1] <= K


def test_factor_prior_without_shocks_is_the_bandit_posterior_635():
    """V of rank zero and one observation per cell: the CRN likelihood
    is plain pulls, so the answer is feature_bandit_posterior's."""
    rng = np.random.default_rng(1)
    K, q = 6, 2
    V0, d0 = rng.normal(size=(K, q)), rng.uniform(0.3, 1.0, K)
    sn = 0.7
    Y = rng.normal(size=(K, 5))
    Y[1, 2:] = np.nan
    Y[4, :] = np.nan
    post = crn_posterior_replicates(np.zeros((K, 0)), sn ** 2, Y,
                                    prior=(V0, d0))
    cnt = (~np.isnan(Y)).sum(1).astype(float)
    ref = feature_bandit_posterior(V0, d0, cnt, np.nansum(Y, 1), sn)
    np.testing.assert_allclose(post[0], ref[0], atol=1e-13)
    np.testing.assert_allclose(post[1], ref[1], rtol=1e-13)
    np.testing.assert_allclose(post[2] @ post[2].T, ref[2] @ ref[2].T,
                               atol=1e-13)


def test_sequential_updates_compose_635(world):
    """The posterior after a first batch, passed back as a factor prior,
    gives the posterior after both batches: the factor prior is the
    form a sequential study chains through."""
    rng, K, rho, R, V, D, Y = world
    s0 = 1.3
    Y1, Y2 = Y[:, :3], Y[:, 3:]
    m1, d1, W1 = crn_posterior_replicates(V, D, Y1, s0)
    seq = crn_posterior_replicates(V, D, Y2, prior=(W1, d1, m1))
    both = crn_posterior_replicates(V, D, Y, s0)
    np.testing.assert_allclose(seq[0], both[0], atol=1e-12)
    np.testing.assert_allclose(np.diag(seq[1]) + seq[2] @ seq[2].T,
                               np.diag(both[1]) + both[2] @ both[2].T,
                               atol=1e-12)


def test_factor_prior_rotates_with_V0_635(world):
    rng, K, rho, R, V, D, Y = world
    V0, d0 = rng.normal(size=(K, 2)), rng.uniform(0.3, 1.0, K)
    Q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    a = crn_posterior_replicates(V, D, Y, prior=(V0, d0))
    b = crn_posterior_replicates(V, D, Y, prior=(V0 @ Q, d0))
    np.testing.assert_allclose(a[0], b[0], atol=1e-12)
    np.testing.assert_allclose(a[2] @ a[2].T, b[2] @ b[2].T, atol=1e-12)


# --------------------------------------------------------------- #638
def test_cell_noise_matrix_is_exact_638(world):
    rng, K, rho, R, V, D, Y = world
    Dc = rng.uniform(0.1, 2.0, (K, R))
    Dc[np.isnan(Y)] = np.nan               # ignored where not observed
    post = crn_posterior_replicates(V, Dc, Y, 0.9)
    ref = _dense(_cells(Y, Dc), 0.81 * np.eye(K), np.zeros(K),
                 [(np.arange(R), V)])
    _close(post, ref)


def test_counts_equal_conditioning_on_every_run_638(world):
    """The cell mean of count_ir runs is sufficient: conditioning on the
    individual runs (sharing their column's shock, independent noise D_i)
    gives the same posterior as the means with counts=."""
    rng, K, rho, R, V, D, Y = world
    cnt = rng.integers(1, 5, (K, R)).astype(float)
    cnt[np.isnan(Y)] = 0
    runs, Ybar = [], np.full((K, R), np.nan)
    for i in range(K):
        for r in range(R):
            if cnt[i, r]:
                x = rng.normal(size=int(cnt[i, r])) + 1.0
                runs += [(i, r, v, D[i]) for v in x]
                Ybar[i, r] = x.mean()
    s0, m0 = 1.1, rng.normal(size=K)
    post = crn_posterior_replicates(V, D, Ybar, s0, m0, counts=cnt)
    ref = _dense(runs, s0 ** 2 * np.eye(K), m0, [(np.arange(R), V)])
    _close(post, ref)
    # and the per-entrant floor the issue had to use instead is NOT it
    floor = D * np.nanmean(np.where(cnt > 0, 1 / np.maximum(cnt, 1), np.nan),
                           axis=1)
    approx = crn_posterior_replicates(V, floor, Ybar, s0, m0)
    assert np.abs(approx[0] - post[0]).max() > 1e-3


# --------------------------------------------------------------- #636
def test_trial_and_seed_shocks_are_exact_636():
    rng = np.random.default_rng(636)
    K, n_trial, n_seed = 7, 4, 3
    trial = np.repeat(np.arange(n_trial), n_seed)
    seed = np.tile(np.arange(n_seed), n_trial)
    R = len(trial)
    Ut, Us, Vc = (rng.normal(size=(K, 2)), rng.normal(size=(K, 1)),
                  0.5 * rng.normal(size=(K, 1)))
    D = rng.uniform(0.2, 1.0, K)
    Y = rng.normal(size=(K, R))
    Y[rng.random((K, R)) < 0.25] = np.nan
    s0 = 1.5
    post = crn_posterior_replicates(Vc, D, Y, s0,
                                    groups=[(trial, Ut), (seed, Us)])
    shocks = [(np.arange(R), Vc), (trial, Ut), (seed, Us)]
    ref = _dense(_cells(Y, np.tile(D[:, None], (1, R))), s0 ** 2 * np.eye(K),
                 np.zeros(K), shocks)
    _close(post, ref)
    # with no column-level shock at all
    post0 = crn_posterior_replicates(None, D, Y, s0,
                                     groups=[(trial, Ut), (seed, Us)])
    ref0 = _dense(_cells(Y, np.tile(D[:, None], (1, R))),
                  s0 ** 2 * np.eye(K), np.zeros(K), shocks[1:])
    _close(post0, ref0)
    # and ignoring the seed level is a different (wrong) posterior
    wrong = crn_posterior_replicates(Vc, D, Y, s0, groups=[(trial, Ut)])
    assert np.abs(wrong[0] - post[0]).max() > 1e-3


def test_duplicate_columns_merge_exactly_with_groups_636():
    """Columns with identical pattern, noise and labels are merged; the
    merge must not merge across different trial or seed labels."""
    rng = np.random.default_rng(6)
    K = 5
    trial = np.array([0, 0, 1, 1, 0, 1])
    seed = np.array([0, 0, 0, 1, 1, 1])
    Ut, Us, Vc = rng.normal(size=(K, 1)), rng.normal(size=(K, 1)), \
        rng.normal(size=(K, 2))
    D = rng.uniform(0.3, 1.0, K)
    Y = rng.normal(size=(K, 6))
    post = crn_posterior_replicates(Vc, D, Y, 1.0,
                                    groups=[(trial, Ut), (seed, Us)])
    ref = _dense(_cells(Y, np.tile(D[:, None], (1, 6))), np.eye(K),
                 np.zeros(K), [(np.arange(6), Vc), (trial, Ut), (seed, Us)])
    _close(post, ref)


@pytest.mark.parametrize("with_prior", [False, True])
def test_free_common_level_is_exact_636(world, with_prior):
    rng, K, rho, R, V, D, Y = world
    Y = Y.copy()
    Y[:, 2] = np.nan
    Y[3, 2] = 0.4                          # a single-entrant column
    m0 = rng.normal(size=K)
    if with_prior:
        V0, d0 = rng.normal(size=(K, 2)), rng.uniform(0.3, 1.0, K)
        kw, S0 = dict(prior=(V0, d0, m0)), V0 @ V0.T + np.diag(d0)
        post = crn_posterior_replicates(V, D, Y, common="free", **kw)
    else:
        S0 = 1.7 ** 2 * np.eye(K)
        post = crn_posterior_replicates(V, D, Y, 1.7, m0, common="free")
    ref = _dense(_cells(Y, np.tile(D[:, None], (1, R))), S0, m0,
                 [(np.arange(R), V)], free=True)
    _close(post, ref, tol=1e-10)


def test_free_common_is_the_limit_of_the_synthetic_column_636(world):
    """The issue's workaround (a common column of large sd) converges to
    common='free' as the sd grows; 'free' is that limit exactly."""
    rng, K, rho, R, V, D, Y = world
    free = crn_posterior_replicates(V, D, Y, 1.0, common="free")
    gaps = []
    for sd in (10.0, 1e3):
        Vs = np.hstack([V, np.full((K, 1), sd)])
        gaps.append(np.abs(crn_posterior_replicates(Vs, D, Y, 1.0)[0]
                           - free[0]).max())
    assert gaps[1] < gaps[0] / 100 and gaps[1] < 1e-4


def test_free_common_ignores_column_levels_636(world):
    """Adding any constant to a whole column changes nothing."""
    rng, K, rho, R, V, D, Y = world
    a = crn_posterior_replicates(V, D, Y, 1.0, common="free")
    b = crn_posterior_replicates(V, D, Y + rng.normal(size=R) * 50, 1.0,
                                 common="free")
    np.testing.assert_allclose(a[0], b[0], atol=1e-10)


def test_all_keywords_together(world):
    rng, K, rho, R, V, D, Y = world
    trial = np.array([0, 0, 1, 1, 2, 2, 3])
    Ut = rng.normal(size=(K, 1))
    V0, d0, m0 = rng.normal(size=(K, 2)), rng.uniform(0.3, 1.0, K), \
        rng.normal(size=K)
    cnt = rng.integers(1, 4, (K, R)).astype(float)
    cnt[np.isnan(Y)] = 0
    post = crn_posterior_replicates(V, D, Y, prior=(V0, d0, m0), counts=cnt,
                                    groups=[(trial, Ut)], common="free")
    Dc = D[:, None] / np.maximum(cnt, 1)
    ref = _dense(_cells(Y, Dc), V0 @ V0.T + np.diag(d0), m0,
                 [(np.arange(R), V), (trial, Ut)], free=True)
    _close(post, ref, tol=1e-10)


def test_scale_equivariance_of_the_general_path(world):
    rng, K, rho, R, V, D, Y = world
    V0, d0 = rng.normal(size=(K, 2)), rng.uniform(0.3, 1.0, K)
    c = 2.0 ** 300
    a = crn_posterior_replicates(V, D, Y, prior=(V0, d0), common="free")
    b = crn_posterior_replicates(V * c, D * c * c, Y * c,
                                 prior=(V0 * c, d0 * c * c), common="free")
    np.testing.assert_allclose(b[0], a[0] * c, rtol=1e-12, atol=1e-12 * c)
    np.testing.assert_allclose(b[1], a[1] * c * c, rtol=1e-12)


# ---------------------------------------------------------- refusals
def test_refusals(world):
    rng, K, rho, R, V, D, Y = world
    with pytest.raises(ValueError, match="not both"):
        crn_posterior_replicates(V, D, Y, 1.0, prior=(V, D))
    with pytest.raises(ValueError, match="s0 is required"):
        crn_posterior_replicates(V, D, Y)
    with pytest.raises(ValueError, match="twice"):
        crn_posterior_replicates(V, D, Y, m0=np.zeros(K),
                                 prior=(V, D, np.zeros(K)))
    with pytest.raises(ValueError, match="singular"):
        crn_posterior_replicates(V, D, Y, prior=V @ V.T)   # rank 2 of 8
    with pytest.raises(ValueError, match="positive semidefinite"):
        crn_posterior_replicates(V, D, Y, prior=-np.eye(K))
    with pytest.raises(ValueError, match="observed cell"):
        Dc = np.ones((K, R))
        Dc[1, 1] = 0.0
        crn_posterior_replicates(V, Dc, Y, 1.0)
    cnt = np.ones((K, R))
    with pytest.raises(ValueError, match="needs a value"):
        crn_posterior_replicates(V, D, Y, 1.0, counts=cnt)
    cnt[np.isnan(Y)] = 0
    cnt[1, 1] = 0
    with pytest.raises(ValueError, match="counts > 0"):
        crn_posterior_replicates(V, D, Y, 1.0, counts=cnt)
    with pytest.raises(ValueError, match="length"):
        crn_posterior_replicates(V, D, Y, 1.0, groups=[(np.arange(3), V)])
    with pytest.raises(ValueError, match="common"):
        crn_posterior_replicates(V, D, Y, 1.0, common="fixed")
    with pytest.raises(ValueError, match="must be"):
        crn_posterior_replicates(V, D, Y, prior=[[1.0, 0.5], [0.5, 1.0]])


def test_no_observations_returns_the_factor_prior(world):
    rng, K, rho, R, V, D, Y = world
    V0, d0, m0 = rng.normal(size=(K, 2)), rng.uniform(0.3, 1.0, K), \
        rng.normal(size=K)
    m, d, W = crn_posterior_replicates(V, D, np.full((K, 3), np.nan),
                                       prior=(V0, d0, m0), common="free")
    np.testing.assert_array_equal(m, m0)
    np.testing.assert_allclose(d, d0, rtol=1e-15)
    np.testing.assert_allclose(W @ W.T, V0 @ V0.T, atol=1e-14)
