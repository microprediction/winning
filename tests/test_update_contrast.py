"""update_contrast (#625): one exact linear-Gaussian observation
a'theta ~ N(mean, var) on a dense or factor belief."""
import numpy as np
import pytest

import winning
import winning.factor as wf
from winning.factor import update_contrast
from winning.factor.core import _center2
from winning.ratings.teams import update_team_margins_full


@pytest.fixture
def belief():
    rng = np.random.default_rng(625)
    K, rho = 8, 2
    V = rng.normal(size=(K, rho))
    D = rng.uniform(0.2, 1.0, K)
    return rng, rng.normal(size=K), V, D, V @ V.T + np.diag(D)


def _precision_form(m, S, a, mean, var):
    """Independent reference: information-form Bayes."""
    P = np.linalg.inv(S)
    P2 = P + np.outer(a, a) / var
    S2 = np.linalg.inv(P2)
    return S2 @ (P @ m + a * mean / var), S2


def test_exported():
    assert wf.update_contrast is winning.update_contrast


def test_dense_equals_information_form(belief):
    rng, m, _, _, S = belief
    a = np.zeros(len(m))
    a[2], a[5] = 1.0, -1.0
    m2, S2 = update_contrast(m, S, a, 0.7, 0.3)
    mr, Sr = _precision_form(m, S, a, 0.7, 0.3)
    np.testing.assert_allclose(m2, mr, atol=1e-12)
    np.testing.assert_allclose(S2, Sr, atol=1e-12)
    a = rng.normal(size=len(m))                    # a general a, too
    m2, S2 = update_contrast(m, S, a, -1.0, 2.0)
    mr, Sr = _precision_form(m, S, a, -1.0, 2.0)
    np.testing.assert_allclose(m2, mr, atol=1e-12)
    np.testing.assert_allclose(S2, Sr, atol=1e-12)


def test_sequential_updates_commute(belief):
    rng, m, _, _, S = belief
    a1, a2 = rng.normal(size=len(m)), rng.normal(size=len(m))
    x = update_contrast(*update_contrast(m, S, a1, 0.3, 0.5), a2, -0.2, 0.8)
    y = update_contrast(*update_contrast(m, S, a2, -0.2, 0.8), a1, 0.3, 0.5)
    np.testing.assert_allclose(x[0], y[0], atol=1e-12)
    np.testing.assert_allclose(x[1], y[1], atol=1e-12)


def test_matches_team_margins_synthetic_observation(belief):
    """The mapping the issue warned about: two one-player teams i, j
    with scores [mean, 0] and beta2 = var/2 observe theta_i - theta_j
    with noise variance var."""
    _, m, _, _, S = belief
    i, j, mean, var = 1, 6, 0.9, 0.4
    a = np.zeros(len(m))
    a[i], a[j] = 1.0, -1.0
    A = np.zeros((2, len(m)))
    A[0, i] = A[1, j] = 1.0
    mt, St, _ = update_team_margins_full(m, S, A, scores=[mean, 0.0],
                                         beta2=var / 2.0)
    m2, S2 = update_contrast(m, S, a, mean, var)
    np.testing.assert_allclose(m2, mt, atol=1e-10)
    np.testing.assert_allclose(S2, St, atol=1e-10)


def test_permutation_equivariance(belief):
    rng, m, _, _, S = belief
    a = rng.normal(size=len(m))
    p = rng.permutation(len(m))
    m2, S2 = update_contrast(m, S, a, 0.1, 0.6)
    mp, Sp = update_contrast(m[p], S[np.ix_(p, p)], a[p], 0.1, 0.6)
    np.testing.assert_allclose(mp, m2[p], atol=1e-13)
    np.testing.assert_allclose(Sp, S2[np.ix_(p, p)], atol=1e-13)


@pytest.mark.parametrize("with_nodes", [False, True])
def test_single_entrant_keeps_the_factor_form_exactly(belief, with_nodes):
    _, m, V, D, S = belief
    a = np.zeros(len(m))
    a[3] = 2.0
    cov = (V, D, np.ones((4, 2)), np.full(4, 0.25)) if with_nodes else (V, D)
    m2, V2, D2, F, W, rep = update_contrast(m, cov, a, 1.5, 0.5)
    mr, Sr = _precision_form(m, S, a, 1.5, 0.5)
    assert rep["exact"] is True and V2.shape == V.shape
    np.testing.assert_allclose(m2, mr, atol=1e-12)
    np.testing.assert_allclose(V2 @ V2.T + np.diag(D2), Sr, atol=1e-12)
    if with_nodes:
        assert F is cov[2] and W is cov[3]
    else:
        assert F.shape[1] == V.shape[1] and W.sum() == pytest.approx(1.0)
    # and the rotation V -> VQ carries through
    Q = np.array([[0.6, -0.8], [0.8, 0.6]])
    _, V3, D3, *_ = update_contrast(m, (V @ Q, D), a, 1.5, 0.5)
    np.testing.assert_allclose(V3, V2 @ Q, atol=1e-13)
    np.testing.assert_allclose(D3, D2, atol=1e-14)


@pytest.mark.parametrize("support", [[0, 4], [1, 2, 6], list(range(8))])
def test_multi_entrant_a_keeps_the_factor_form_exactly(belief, support):
    """Exact at rank rho + |support| - 1, with a positive diagonal."""
    rng, m, V, D, S = belief
    a = np.zeros(len(m))
    a[support] = rng.normal(size=len(support))
    m2, V2, D2, F, W, rep = update_contrast(m, (V, D), a, 0.4, 0.3)
    mr, Sr = _precision_form(m, S, a, 0.4, 0.3)
    assert rep == {"exact": True, "rank": V.shape[1] + len(support) - 1}
    assert V2.shape == (len(m), rep["rank"]) and (D2 > 0).all()
    np.testing.assert_allclose(m2, mr, atol=1e-12)
    np.testing.assert_allclose(V2 @ V2.T + np.diag(D2), Sr, atol=1e-12)
    np.testing.assert_allclose(np.delete(D2, support),
                               np.delete(D, support))
    assert F.shape[1] == rep["rank"] and len(F) == len(W)


def test_sequential_factor_updates_match_dense(belief):
    rng, m, V, D, S = belief
    obs = []
    for _ in range(3):
        a = np.zeros(len(m))
        i, j = rng.choice(len(m), 2, replace=False)
        a[i], a[j] = 1.0, -1.0
        obs.append((a, rng.normal(), 0.5))
    md, Sd, mf, cov = m, S, m, (V, D)
    for a, y, v in obs:
        md, Sd = update_contrast(md, Sd, a, y, v)
        out = update_contrast(mf, cov, a, y, v)
        mf, cov = out[0], (out[1], out[2])
    assert cov[0].shape[1] == V.shape[1] + 3
    np.testing.assert_allclose(mf, md, atol=1e-11)
    np.testing.assert_allclose(cov[0] @ cov[0].T + np.diag(cov[1]), Sd,
                               atol=1e-11)


def test_refit_option_reports_its_residual(belief):
    _, m, V, D, S = belief
    a = np.zeros(len(m))
    a[0], a[4] = 1.0, -1.0
    m2, V2, D2, F, W, rep = update_contrast(m, (V, D), a, 0.4, 0.3,
                                            refit=True)
    mr, Sr = _precision_form(m, S, a, 0.4, 0.3)
    np.testing.assert_allclose(m2, mr, atol=1e-12)    # mean is exact
    assert rep["exact"] is False
    for key in ("projected_residual_rel", "projected_residual_max",
                "contrast_residual_max", "rank"):
        assert key in rep
    R = _center2(Sr - V2 @ V2.T - np.diag(D2))
    assert float(np.abs(R).max() / np.mean(np.diag(Sr))) == pytest.approx(
        rep["projected_residual_max"], rel=1e-6, abs=1e-12)
    assert len(F) == len(W)


def test_orthogonal_contrast_does_not_preserve_the_form(belief):
    """The tempting special case, checked: a orthogonal to every
    loading leaves g = D a, and the downdate g g' is still off-diagonal
    when a touches two entrants, so it is not a D-only change."""
    _, m, V, D, S = belief
    a = np.zeros(len(m))
    a[0], a[1] = V[1, 0], -V[0, 0]
    V0 = V.copy()
    V0[:, 1] = 0.0
    assert abs(V0.T @ a).max() < 1e-14
    S0 = V0 @ V0.T + np.diag(D)
    _, S2 = update_contrast(m, S0, a, 0.0, 1.0)
    delta = S0 - S2
    assert abs(delta[0, 1]) > 1e-3


def test_validation(belief):
    _, m, V, D, S = belief
    a = np.ones(len(m))
    for bad_var in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            update_contrast(m, S, a, 0.0, bad_var)
        with pytest.raises(ValueError):
            update_contrast(m, (V, D), a, 0.0, bad_var)
    with pytest.raises(ValueError):
        update_contrast(m, S, np.zeros(len(m)), 0.0, 1.0)
    with pytest.raises(ValueError):
        update_contrast(m, S, np.r_[np.nan, a[1:]], 0.0, 1.0)
    with pytest.raises(ValueError):
        update_contrast(m, S, a[1:], 0.0, 1.0)
    with pytest.raises(ValueError):
        update_contrast(m, S, a, np.nan, 1.0)
    with pytest.raises(ValueError):
        update_contrast(m, (V, D, None), a, 0.0, 1.0)
    with pytest.raises(ValueError):
        update_contrast(m, S[:-1], a, 0.0, 1.0)
