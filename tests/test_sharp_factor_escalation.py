"""Regression: sharp factor loadings escalate to a low-discrepancy
family, not a deeper Gauss-Hermite rule.

Found by the adversarial battery behind the general-inversion paper
(docs/latex_src/general_inversion/break.py, section H): at implied
correlations beyond ~0.9 the conditional race is a near-step in factor
space, Gauss-Hermite converges slowly at any order (the 25-node rule
still lost ~1e-2 TV), and the default silently returned percent-level
errors. The default now switches to scrambled-Sobol nodes past
sharpness 3; this test pins the switched regime to reference accuracy
and the mild regime to its previous behavior.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from winning.factor.races import race_probabilities


def _reference(mu, V, D, m=19):
    L = np.linalg.cholesky(V @ V.T + np.diag(D))
    z = ndtri(np.clip(qmc.Sobol(len(mu), scramble=True, seed=1)
                      .random_base2(m), 1e-12, 1 - 1e-12)).T
    return np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                       minlength=len(mu)) / z.shape[1]


def test_sharp_factor_default_hits_reference_noise():
    rng = np.random.default_rng(99)
    n = 20
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 4.0          # sharpness ~ 10
    D = 0.5 + rng.random(n)
    ref = _reference(mu, V, D)
    p = race_probabilities(mu, V=V, D=D, points=513)
    tv = 0.5 * np.abs(p - ref).sum()
    # pre-escalation default measured 1.25e-2 here; reference noise at
    # 2^19 is ~6e-4, so 2.5e-3 fails the old behavior with margin
    assert tv < 2.5e-3, f"sharp-factor TV {tv:.2e}"


def test_mild_factor_unchanged():
    rng = np.random.default_rng(99)
    n = 20
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 0.4          # sharpness ~ 1
    D = 0.5 + rng.random(n)
    ref = _reference(mu, V, D)
    p = race_probabilities(mu, V=V, D=D, points=513)
    tv = 0.5 * np.abs(p - ref).sum()
    assert tv < 2.5e-3, f"mild-factor TV {tv:.2e}"
