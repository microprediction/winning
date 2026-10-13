"""fit_factor_model_missing (#637): maximum-likelihood (V, D) by EM from
a trials x entrants score matrix with NaN cells. Recovery is tested on
simulated data with known factors under two stated missingness patterns:
MCAR (each cell missing independently with probability 0.3) and a
staggered design (each trial scores a window of 6 consecutive entrants
out of 10, so some pairs are never scored together)."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import winning
import winning.factor as wf
from winning.factor import crn_posterior_replicates, fit_factor_model_missing


def _simulate(seed, K=10, k=2, T=4000):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(K, k))
    D = rng.uniform(0.3, 1.0, K)
    mu = rng.normal(size=K)
    Y = mu + rng.normal(size=(T, k)) @ V.T + rng.normal(size=(T, K)) * \
        np.sqrt(D)
    return rng, V, D, mu, Y


def _mcar(rng, Y, p=0.3):
    Ym = Y.copy()
    Ym[rng.random(Y.shape) < p] = np.nan
    return Ym


def _staggered(rng, Y, width=6):
    T, K = Y.shape
    Ym = np.full_like(Y, np.nan)
    for t in range(T):
        s = rng.integers(0, K - width + 1)
        Ym[t, s:s + width] = Y[t, s:s + width]
    return Ym


def _rel(A, B):
    return float(np.abs(A - B).max() / np.abs(B).max())


def test_exported():
    assert wf.fit_factor_model_missing is winning.fit_factor_model_missing


def test_recovers_known_factors_under_mcar():
    rng, V, D, mu, Y = _simulate(637)
    Vh, Dh, rep = fit_factor_model_missing(_mcar(rng, Y), 2)
    assert rep["converged"]
    assert _rel(Vh @ Vh.T, V @ V.T) < 0.08
    assert np.abs(Dh / D - 1).max() < 0.15
    assert np.abs(rep["mean"] - mu).max() < 0.08
    assert rep["missing_fraction"] == pytest.approx(0.3, abs=0.01)


def test_recovers_known_factors_when_pairs_are_never_scored_together():
    rng, V, D, mu, Y = _simulate(638, k=1)
    Ym = _staggered(rng, Y)
    with pytest.raises(ValueError, match="min_pairs"):
        fit_factor_model_missing(Ym, 1)
    Vh, Dh, rep = fit_factor_model_missing(Ym, 1, min_pairs=0)
    assert rep["pairs"][0, 9] == 0 and rep["min_pairs"] == 0
    assert rep["converged"]
    # including the never-observed (0, 9) entry, carried by the factor
    assert _rel(Vh @ Vh.T, V @ V.T) < 0.1
    assert np.abs(Dh / D - 1).max() < 0.2


def test_error_shrinks_with_more_trials():
    errs = []
    for T in (500, 8000):
        rng, V, D, mu, Y = _simulate(1, T=T)
        Vh, Dh, _ = fit_factor_model_missing(_mcar(rng, Y), 2)
        errs.append(_rel(Vh @ Vh.T, V @ V.T))
    assert errs[1] < errs[0] / 2


def test_em_is_monotone_and_the_loglik_is_exact():
    rng, V, D, mu, Y = _simulate(2, T=300)
    Ym = _mcar(rng, Y)
    Vh, Dh, rep = fit_factor_model_missing(Ym, 2)
    tr = rep["trace"]
    assert (np.diff(tr) >= -1e-9 * abs(tr[-1])).all()
    S = Vh @ Vh.T + np.diag(Dh)
    ll = sum(multivariate_normal(rep["mean"][o], S[np.ix_(o, o)]).logpdf(r[o])
             for r in Ym for o in [~np.isnan(r)])
    assert rep["loglik"] == pytest.approx(ll, rel=1e-12)
    # an MLE: at least the likelihood of the truth
    St = V @ V.T + np.diag(D)
    ll_true = sum(multivariate_normal(mu[o], St[np.ix_(o, o)]).logpdf(r[o])
                  for r in Ym for o in [~np.isnan(r)])
    assert rep["loglik"] >= ll_true


def test_complete_data_matches_the_closed_form_at_k0():
    rng, V, D, mu, Y = _simulate(3, T=200)
    Vh, Dh, rep = fit_factor_model_missing(Y, 0)
    assert Vh.shape == (10, 0)
    np.testing.assert_allclose(rep["mean"], Y.mean(0), rtol=1e-12)
    np.testing.assert_allclose(Dh, Y.var(0), rtol=1e-10)


def test_bic_chooses_the_true_rank():
    rng, V, D, mu, Y = _simulate(4, T=1000)
    Vh, Dh, rep = fit_factor_model_missing(_mcar(rng, Y), k_max=3,
                                           max_iter=20000)
    assert rep["k"] == 2 and set(rep["bic"]) == {0, 1, 2, 3}
    assert all(rep["converged_by_k"].values())


def test_deterministic_given_seed():
    rng, V, D, mu, Y = _simulate(5, T=400)
    Ym = _mcar(rng, Y)
    a = fit_factor_model_missing(Ym, 2, n_starts=3, seed=7)
    b = fit_factor_model_missing(Ym, 2, n_starts=3, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    # V is canonical: V' D^-1 V diagonal and decreasing
    G = (a[0].T / a[1]) @ a[0]
    assert abs(G[0, 1]) < 1e-8 * G[0, 0] and G[0, 0] >= G[1, 1]


def test_feeds_the_crn_posterior():
    rng, V, D, mu, Y = _simulate(6, T=300)
    Ym = _mcar(rng, Y)
    Vh, Dh, _ = fit_factor_model_missing(Ym, 2)
    m, d, W = crn_posterior_replicates(Vh, Dh, Ym.T, 3.0)
    assert np.isfinite(m).all() and (d > 0).all()


def test_nonconvergence_is_reported():
    rng, V, D, mu, Y = _simulate(7, T=200)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        _, _, rep = fit_factor_model_missing(_mcar(rng, Y), 2, max_iter=3)
    assert not rep["converged"]


def test_refusals():
    rng, V, D, mu, Y = _simulate(8, T=50)
    with pytest.raises(ValueError, match="Ledermann"):
        fit_factor_model_missing(Y, 7)
    Ym = Y.copy()
    Ym[2:, 3] = np.nan
    with pytest.raises(ValueError, match="entrant 3"):
        fit_factor_model_missing(Ym, 1, min_pairs=0)
    with pytest.raises(ValueError, match="infinite"):
        fit_factor_model_missing(np.where(Y > 2, np.inf, Y), 1)
    with pytest.raises(ValueError, match="non-negative integer"):
        fit_factor_model_missing(Y, 1.5)
