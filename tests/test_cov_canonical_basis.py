"""cov= prices do not depend on the eigensolver's sign choices (#745).

fit_covariance's V is identified only up to V -> V O, and LAPACK picks
each eigenvector's sign: macOS and Linux fitted the n6r3 sample
covariance with one column negated, and the scrambled-Sobol nodes are
not sign symmetric, so race_probabilities(mu, cov=S) differed by 6e-5
between platforms. The basis is now canonical (eigenbasis of V'V,
largest-|entry| of each column positive) before the nodes are built.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import winning.factor.core as core
from winning.factor.core import _canonical_factor_basis, fit_covariance
from winning.factor.races import _fit_cov, race_probabilities

sys.path.insert(0, str(Path(__file__).parent))
from ordinary_battery import _sample_covariances  # noqa: E402


def _price(mu, S):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return race_probabilities(mu, cov=S)


def _with_eigvec_signs(signs):
    """_top_eigen with its eigenvector columns negated by `signs`."""
    orig = core._top_eigen

    def flipped(M, k, *a, **kw):
        lam, U, Q = orig(M, k, *a, **kw)
        return lam, U * signs[:U.shape[1]], Q
    return orig, flipped


def _n6r3():
    return dict(_sample_covariances())["n6r3"]


def test_canonical_basis_keeps_the_covariance_and_is_idempotent():
    V, _D, _F, _W = fit_covariance(_n6r3())
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for X in (V, V @ Q, -V):
        C = _canonical_factor_basis(X)
        np.testing.assert_allclose(C @ C.T, V @ V.T, rtol=0, atol=1e-14)
        np.testing.assert_allclose(_canonical_factor_basis(C), C,
                                   rtol=0, atol=1e-15)


@pytest.mark.parametrize("signs", list(itertools.product([1.0, -1.0],
                                                          repeat=3)))
def test_n6r3_all_sign_patterns_give_one_basis(signs):
    V, _D, _F, _W = fit_covariance(_n6r3())
    s = np.array(signs)
    assert np.array_equal(_canonical_factor_basis(V * s),
                          _canonical_factor_basis(V))
    # and a column permutation is the same basis to rounding
    np.testing.assert_allclose(_canonical_factor_basis((V * s)[:, ::-1]),
                               _canonical_factor_basis(V), rtol=0, atol=1e-14)


def test_n6r3_price_ignores_eigensolver_signs(monkeypatch):
    # every sign pattern LAPACK could return, at every eigen call in the
    # fit, prices the same race bit for bit; main moved 6e-5 to 1.6e-4
    S, mu = _n6r3(), np.linspace(-1.0, 1.0, 6)
    want = _price(mu, S)
    for signs in itertools.product([1.0, -1.0], repeat=6):
        orig, flipped = _with_eigvec_signs(np.array(signs))
        monkeypatch.setattr(core, "_top_eigen", flipped)
        assert np.array_equal(_price(mu, S), want), signs
        monkeypatch.setattr(core, "_top_eigen", orig)


@pytest.mark.parametrize("driver", ["evd", "evr", "ev", "evx"])
def test_n6r3_price_ignores_the_eigh_driver(monkeypatch, driver):
    # scipy's evr flipped a column on main and moved the price 1.06e-4;
    # drivers differ in rounding only now
    S, mu = _n6r3(), np.linspace(-1.0, 1.0, 6)
    want = _price(mu, S)
    monkeypatch.setattr(core.np.linalg, "eigh",
                        lambda M: scipy.linalg.eigh(M, driver=driver))
    np.testing.assert_allclose(_price(mu, S), want, rtol=0, atol=1e-13)


def _factor_route_covs():
    """Ordinary covariances whose cov= call prices the fit (not GHK):
    sample covariances of 50n draws and exact V V' + diag(d). (At 4n
    draws most fits degrade and route to dense GHK, which has no basis.)"""
    rng = np.random.default_rng(745688)
    out = []
    for kind in ("sample", "exact"):
        for n in (5, 6, 14, 20, 30):
            for r in (1, 2, 3):
                V = rng.normal(size=(n, r)) * rng.uniform(0.4, 1.2)
                d = rng.uniform(0.3, 1.5, size=n)
                if kind == "exact":
                    S = V @ V.T + np.diag(d)
                else:
                    X = (rng.standard_normal((50 * n, r)) @ V.T
                         + rng.standard_normal((50 * n, n)) * np.sqrt(d))
                    S = np.cov(X.T)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if not _fit_cov(S, None, None, None,
                                    will_route=True)[4]:
                        out.append((f"{kind}_n{n}r{r}", S))
    return out


FACTOR_COVS = _factor_route_covs()


def test_the_battery_exercises_the_factor_route():
    assert len(FACTOR_COVS) >= 15


@pytest.mark.parametrize("label,S", FACTOR_COVS,
                         ids=[c[0] for c in FACTOR_COVS])
def test_ordinary_cov_prices_ignore_eigensolver_choices(monkeypatch,
                                                        label, S):
    mu = np.linspace(-1.0, 1.0, len(S))
    want = _price(mu, S)
    rng = np.random.default_rng(len(S))
    for _ in range(4):
        signs = rng.choice([-1.0, 1.0], size=len(S))
        orig, flipped = _with_eigvec_signs(signs)
        monkeypatch.setattr(core, "_top_eigen", flipped)
        # equal to rounding, not always bitwise: a pipeline fit's columns
        # are not orthogonal, so the flips change V'V's off-diagonal signs
        # and its eigh rounds differently (seen: 1.1e-16). Main moved
        # these by QMC error, ~1e-4.
        np.testing.assert_allclose(_price(mu, S), want, rtol=0, atol=1e-14,
                                   err_msg=label)
        monkeypatch.setattr(core, "_top_eigen", orig)
    # another driver rounds differently, and the fit's alternation carries
    # that to ~1e-11 (Linux CI: 1.0e-11 on sample_n20r3; the drift guard's
    # FIT_ATOL exists for the same reason). A sign or basis change, as on
    # main, moves the price by ~1e-4.
    monkeypatch.setattr(core.np.linalg, "eigh",
                        lambda M: scipy.linalg.eigh(M, driver="evr"))
    np.testing.assert_allclose(_price(mu, S), want, rtol=0, atol=1e-9)
