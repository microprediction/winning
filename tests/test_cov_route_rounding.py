"""The cov= route decision does not hinge on rounding (#745).

race_probabilities(mu, cov=S) prices a healthy factor fit by its nodes
and routes a degraded one (rank >= n, idiosyncratic floor bound, or a
residual over 0.05) to dense GHK. #745 suspected that decision of
flipping between platforms on an ordinary sample covariance. It does
not: on every ordinary sample covariance below, perturbing S at 1e-14
relative leaves the fit's rank, its floor count, its residuals' side of
0.05 and the degraded flag unchanged, and the routed answer moves only
at rounding. (The cross-platform difference #745 found comes from the
sign of a fitted loading column -- see test_default_drift.py.)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from winning.factor.core import fit_covariance
from winning.factor.races import _fit_cov, race_probabilities

sys.path.insert(0, str(Path(__file__).parent))
from ordinary_battery import _sample_covariances  # noqa: E402


def _covs():
    out = list(_sample_covariances())
    rng = np.random.default_rng(745)
    for n in (4, 6, 8, 10, 16):
        for r in (1, 2, 3):
            V = rng.normal(size=(n, r)) * rng.uniform(0.4, 1.2)
            d = rng.uniform(0.3, 1.5, size=n)
            m = 4 * n
            X = (rng.standard_normal((m, r)) @ V.T
                 + rng.standard_normal((m, n)) * np.sqrt(d))
            out.append((f"extra_n{n}r{r}", np.cov(X.T)))
    return out


def _decision(S):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _V, D, _F, _W, rep = fit_covariance(S, return_report=True)
        degraded = _fit_cov(S, None, None, None, will_route=True)[4]
    diag = np.diag(S)
    floor = 1e-6 * np.maximum(diag, 1e-6 * diag.mean())
    return (rep["rank"], int((D <= 2.0 * floor).sum()),
            rep.get("contrast_residual_max", 0.0) > 0.05,
            rep["projected_residual_max"] > 0.05, degraded)


@pytest.mark.parametrize("label,S", _covs(), ids=[c[0] for c in _covs()])
def test_route_decision_is_invariant_under_rounding(label, S):
    want = _decision(S)
    mu = np.linspace(-1.0, 1.0, len(S))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = race_probabilities(mu, cov=S)
    rng = np.random.default_rng(14)
    for _ in range(5):
        E = rng.normal(size=S.shape)
        S2 = S * (1.0 + 1e-14 * (E + E.T) / 2.0)
        assert _decision(S2) == want, label
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p2 = race_probabilities(mu, cov=S2)
        # same route, same nodes: the answer moves at rounding, far below
        # the 6e-5 a route change (or a loading-sign flip) produces
        np.testing.assert_allclose(p2, p, rtol=0, atol=1e-10)
