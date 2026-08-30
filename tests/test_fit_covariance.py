"""The dense-covariance front door: race_probabilities(mu, cov=C).

Goldens here are measured, not guessed: the MC referees below were run at
4e5 draws (TV noise ~1e-3 at n=40) when the thresholds were set.
"""
import numpy as np
import pytest

from winning import race_probabilities
from winning.factor.core import fit_covariance, qmc_nodes


def _mc_tv(mu, C, p, draws=400_000, seed=123):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C + 1e-10 * np.eye(len(C)))
    counts = np.bincount(
        (mu + rng.standard_normal((draws, len(C))) @ L.T).argmin(1),
        minlength=len(C)) / draws
    return 0.5 * np.abs(counts - p).sum()


def test_in_grammar_truth_recovered():
    # C already factor+diagonal: the fit must not distort it. The old
    # pipeline (blocks chasing the raw residual's choice-irrelevant common
    # component) scored TV 0.017 here; the projected-residual pipeline
    # measures 1.9e-4, which is lattice/node noise.
    rng = np.random.default_rng(0)
    n = 40
    Vt = rng.normal(size=(n, 2)) * 0.6
    Dt = 0.5 + rng.random(n)
    C = Vt @ Vt.T + np.diag(Dt)
    mu = np.sort(rng.normal(size=n))
    F, W = qmc_nodes(2, 11)
    p_true = race_probabilities(mu, V=Vt, D=Dt, F=F, W=W, points=257)
    p_fit = race_probabilities(mu, cov=C, points=257)
    assert 0.5 * np.abs(p_true - p_fit).sum() < 1e-3


def test_common_component_is_ignored():
    # adding tau^2 11' to Sigma cannot move any win probability; the fit
    # must agree with itself across tau (the identification fact the
    # projected objective encodes).
    rng = np.random.default_rng(1)
    n = 30
    Vt = rng.normal(size=(n, 2)) * 0.5
    C = Vt @ Vt.T + np.diag(0.5 + rng.random(n))
    mu = np.sort(rng.normal(size=n))
    p0 = race_probabilities(mu, cov=C, points=257)
    p9 = race_probabilities(mu, cov=C + 9.0 * np.ones((n, n)), points=257)
    assert 0.5 * np.abs(p0 - p9).sum() < 2e-3


def test_equicorrelation_matches_mc():
    rng = np.random.default_rng(2)
    n = 40
    C = np.full((n, n), 0.6)
    np.fill_diagonal(C, 1.0)
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, cov=C)
    assert abs(p.sum() - 1.0) < 1e-9
    assert _mc_tv(mu, C, p) < 5e-3   # measured 1.1e-3


def test_ar1_within_measured_band():
    # AR(1) is the family the grammar serves worst (locality); measured
    # TV 0.049 at rho=0.9, n=40. This is a regression rail, not a claim
    # of exactness.
    rng = np.random.default_rng(0)
    n = 40
    C = 0.9 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, cov=C)
    assert _mc_tv(mu, C, p) < 0.08


def test_cov_excludes_other_specs():
    with pytest.raises(ValueError):
        race_probabilities([0.0, 1.0], cov=np.eye(2), D=np.ones(2))


def test_grammar_inversion_round_trips():
    # the intro table's claim: priced and inverted by the same two calls,
    # for every grammar member (generic damped inverse, exact forward map)
    from winning.factor.structures import (Blocks, Nested, Tree,
                                           dispatch_probabilities)
    from winning.factor.races import abilities_from_race
    rng = np.random.default_rng(3)
    n = 60
    cluster = np.repeat(np.arange(6), 10)
    mu0 = rng.normal(size=n)
    mu0 -= mu0.mean()
    parent = np.array([6, 6, 7, 7, 8, 8, 8, 8, -1])
    strength = np.concatenate([np.zeros(6), [0.4, 0.4, 0.3]])
    cases = [
        Blocks(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
               D=0.5 + rng.random(n)),
        Nested(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
               D=0.5 + rng.random(n), coupling=0.5 * rng.random(n),
               gamma=0.8),
        Tree(cluster=cluster, loading=0.4 + 0.3 * rng.random(n),
             D=0.5 + rng.random(n), parent=parent, strength=strength),
    ]
    for s in cases:
        pstar = dispatch_probabilities(mu0, s)
        mu_hat = abilities_from_race(pstar, structure=s)
        assert np.abs(mu_hat - mu0).max() < 1e-6


def test_cov_inversion_round_trip():
    # invert the fitted race: forward with cov=, invert with cov=,
    # recover the abilities of the fitted model
    rng = np.random.default_rng(4)
    n = 30
    Vt = rng.normal(size=(n, 2)) * 0.5
    C = Vt @ Vt.T + np.diag(0.5 + rng.random(n))
    mu0 = rng.normal(size=n)
    mu0 -= mu0.mean()
    from winning.factor.races import abilities_from_race
    p = race_probabilities(mu0, cov=C)
    mu_hat = abilities_from_race(p, cov=C)
    assert np.abs(mu_hat - mu0).max() < 1e-5


def test_hard_covariance_warns_easy_does_not():
    # the front door tells the user when the grammar fit is imperfect
    import warnings
    rng = np.random.default_rng(0)
    n = 40
    mu = np.sort(rng.normal(size=n))
    Vt = rng.normal(size=(n, 2)) * 0.5
    C_easy = Vt @ Vt.T + np.diag(0.5 + rng.random(n))
    C_hard = 0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        race_probabilities(mu, cov=C_easy)
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        race_probabilities(mu, cov=C_hard)
        assert any("grammar fit" in str(x.message) for x in w)


def test_bad_covariances_raise_clearly():
    # gap-stress find: a non-PSD "covariance" silently priced to all-NaN
    # probabilities, and a NaN entry died inside LAPACK with an
    # inscrutable message; both now raise at the door with a diagnosis
    rng = np.random.default_rng(0)
    mu = np.linspace(-1, 1, 6)
    A = rng.normal(size=(6, 6))
    C_npsd = 0.3 * (A + A.T) / 2 + np.eye(6) * 0.3 - 0.8 * np.eye(6)
    with pytest.raises(ValueError, match="positive semidefinite"):
        race_probabilities(mu, cov=C_npsd)
    C_nan = np.eye(6)
    C_nan[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        race_probabilities(mu, cov=C_nan)
    C_asym = np.eye(6) + 0.3 * rng.normal(size=(6, 6))
    with pytest.raises(ValueError, match="symmetric"):
        race_probabilities(mu, cov=C_asym)
    # a singular but valid covariance (rank 2) must still price
    B = rng.normal(size=(6, 2))
    p = race_probabilities(mu, cov=B @ B.T)
    assert abs(p.sum() - 1) < 1e-9 and np.isfinite(p).all()


def test_heterogeneous_scales_analytic_two_alternative():
    # fourth review's counterexample: C equicorrelated (rho = 0.9),
    # S = diag(1, 2). A CORRELATION-space quotient fit that restores
    # scales afterwards gets the only contrast that matters wrong
    # (difference variance 0.5 instead of 1.4); the shipped fit works in
    # covariance coordinates and must match the analytic binary
    # probability. (Their example also exposed an n=2 division by zero
    # in the water-filling diagonal solve, now guarded.)
    from scipy.stats import norm
    rho = 0.9
    C = np.array([[1.0, rho], [rho, 1.0]])
    S = np.diag([1.0, 2.0])
    Sigma = S @ C @ S
    mu = np.array([0.0, 0.5])
    dvar = Sigma[0, 0] + Sigma[1, 1] - 2 * Sigma[0, 1]
    p1 = norm.cdf((mu[1] - mu[0]) / np.sqrt(dvar))
    p = race_probabilities(mu, cov=Sigma)
    assert abs(p[0] - p1) < 1e-6


def test_exact_grammar_round_trip_with_unequal_variances():
    # variances spanning two decades; supplying Sigma through the dense
    # front door must agree with supplying (V, D) directly
    import warnings
    rng = np.random.default_rng(0)
    n = 12
    V = rng.normal(size=(n, 2)) * 0.6
    D = np.exp(rng.uniform(np.log(0.1), np.log(10), n))
    Sigma = V @ V.T + np.diag(D)
    mu = np.sort(rng.normal(size=n))
    F, W = qmc_nodes(2, 12)
    p_direct = race_probabilities(mu, V=V, D=D, F=F, W=W, points=513)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_dense = race_probabilities(mu, cov=Sigma, points=513)
    assert 0.5 * np.abs(p_direct - p_dense).sum() < 2e-2
