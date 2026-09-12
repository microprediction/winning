"""Factor-correlated ranked/winner moment updates: the ratings layer's
last open item. Verified the only way that counts: rejection-sampled
Monte Carlo posteriors of the generative model."""
import numpy as np

from winning.ratings.nway import (update_order_correlated, update_ranking_exact,
                                  update_winner, update_winner_correlated)


from winning.ratings.simulate import correlated_draws as _simulate  # noqa: E402


def test_winner_update_matches_mc_posterior():
    rng = np.random.default_rng(0)
    m = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
    v = np.array([0.8, 1.0, 0.6, 1.2, 0.9])
    V = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])
    beta2 = 1.0
    s, X = _simulate(rng, 2_000_000, m, v, V, beta2)
    keep = X.argmax(axis=1) == 3
    s_win = s[keep]
    m_mc, v_mc = s_win.mean(axis=0), s_win.var(axis=0)
    m_hat, v_hat, logZ = update_winner_correlated(m, v, 3, V, beta2)
    se = np.sqrt(v_mc / keep.sum())
    assert np.abs(m_hat - m_mc).max() < 4 * se.max() + 2e-3
    assert np.abs(v_hat - v_mc).max() < 0.01
    assert abs(np.exp(logZ) - keep.mean()) < 3 * np.sqrt(0.25 / len(s))
    # correlation matters: the independent update is measurably wrong here
    m_ind, _, _ = update_winner(m, v, 3, beta2)
    assert np.abs(m_ind - m_mc).max() > 4 * np.abs(m_hat - m_mc).max()


def test_order_update_matches_mc_posterior():
    rng = np.random.default_rng(1)
    m = np.array([0.4, 0.0, -0.3, 0.0])
    v = np.array([0.9, 0.9, 0.9, 0.9])
    V = np.array([[0.9], [0.8], [-0.7], [0.1]])
    beta2 = 1.0
    s, X = _simulate(rng, 4_000_000, m, v, V, beta2)
    order = np.array([1, 0, 3, 2])
    ranks = np.argsort(-X, axis=1)
    keep = (ranks == order).all(axis=1)
    s_o = s[keep]
    m_mc, v_mc = s_o.mean(axis=0), s_o.var(axis=0)
    m_hat, v_hat, logZ = update_order_correlated(m, v, order, V, beta2)
    se = np.sqrt(v_mc / keep.sum())
    assert np.abs(m_hat - m_mc).max() < 4 * se.max() + 3e-3
    assert np.abs(v_hat - v_mc).max() < 0.015
    assert abs(np.exp(logZ) - keep.mean()) < 3 * np.sqrt(0.05 / len(s))
    # the shared-realization-correct independent update is biased under
    # correlation; the factor mixture must beat it clearly
    m_ind, _ = update_ranking_exact(m, v, order, beta2)
    assert np.abs(m_ind - m_mc).max() > 2 * np.abs(m_hat - m_mc).max()


def test_v_zero_reduces_to_independent_members():
    m = np.array([0.2, 0.0, -0.1, 0.3])
    v = np.ones(4) * 0.8
    V0 = np.zeros((4, 1))
    mw, vw, lz = update_winner_correlated(m, v, 2, V0)
    mw0, vw0, p0 = update_winner(m, v, 2, beta2=1.0)
    assert np.abs(mw - mw0).max() < 1e-12
    assert abs(lz - np.log(p0)) < 1e-12
    mo, vo, _ = update_order_correlated(m, v, [2, 0, 1, 3], V0)
    mo0, vo0 = update_ranking_exact(m, v, [2, 0, 1, 3])
    assert np.abs(mo - mo0).max() < 1e-12
    assert np.abs(vo - vo0).max() < 1e-10


def test_impossible_order_degrades_gracefully():
    m = np.linspace(4, -4, 5)
    v = np.full(5, 0.01)
    V = np.full((5, 1), 0.3)
    m_hat, v_hat, logZ = update_order_correlated(m, v, [4, 3, 2, 1, 0], V,
                                                 beta2=0.01)
    assert np.isfinite(m_hat).all() and np.isfinite(v_hat).all()
    assert logZ < -50


def test_laplace_updates_shrink_variance():
    """The laplace density's kink leaves O(dx) noise on the lattice
    gradient; differencing that gradient at eps=1e-3 amplified it into
    wrong-signed curvature, and forty repeated order updates inflated a
    unit prior variance to 18 (bandits harness, 2026-09-01). Log-concave
    bases have concave log-evidence in the means, so variance must be
    non-increasing; the fix widens the differencing step for laplace and
    clamps positive curvature for the named log-concave bases."""
    import warnings
    from winning.ratings import update_order_correlated

    rng = np.random.default_rng(0)
    M = 10
    a = rng.normal(0, 1, M)
    a -= a.mean()
    mean = np.zeros(M)
    var = np.ones(M)
    V = np.zeros((M, 1))
    r2 = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(25):
            S = r2.choice(M, 5, replace=False)
            perf = a[S] + r2.laplace(0, 1 / np.sqrt(2), 5)
            order = np.argsort(perf)          # min-wins order, first to last
            m_s, v_s, _ = update_order_correlated(mean[S], var[S], order,
                                                  V[S], base="laplace")
            mean[S] = m_s
            var[S] = v_s
    assert var.max() <= 1.0 + 1e-9, var.max()
    assert var.mean() < 0.35, var.mean()
    assert np.corrcoef(-mean, a)[0, 1] > 0.8


def test_node_count_parameters_apply_where_documented():
    # rank >= 3: Qf is inert (it was silently ignored: the Sobol count
    # was hard-coded at 1024), nodes_log2 is the knob, and the default
    # reproduces the historical count
    rng = np.random.default_rng(4)
    m = rng.normal(size=4); v = np.full(4, 0.9)
    V = rng.normal(0, 0.4, (4, 3))
    a = update_winner_correlated(m, v, 1, V, Qf=3)
    b = update_winner_correlated(m, v, 1, V, Qf=7)
    assert np.abs(a[0] - b[0]).max() == 0.0 and a[2] == b[2]
    c = update_winner_correlated(m, v, 1, V, nodes_log2=8)
    d = update_winner_correlated(m, v, 1, V)
    e = update_winner_correlated(m, v, 1, V, nodes_log2=10)
    assert np.abs(c[0] - d[0]).max() > 0.0                # the knob acts
    assert np.abs(c[0] - d[0]).max() < 2e-2               # and converges
    assert np.abs(d[0] - e[0]).max() == 0.0 and d[2] == e[2]
    # rank <= 2: Qf is the knob
    V2 = V[:, :2]
    f = update_winner_correlated(m, v, 1, V2, Qf=3)
    g = update_winner_correlated(m, v, 1, V2, Qf=7)
    assert np.abs(f[0] - g[0]).max() > 0.0
