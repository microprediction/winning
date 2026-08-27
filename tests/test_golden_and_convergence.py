"""Golden-board regression, convergence-order, statistical-MC and KKT tests.

The golden boards (tests/golden/race_boards.npz) were computed on the numpy
path at very high resolution (points up to 4097, qa up to 21, qf 31) from a
fixed seed; production settings must reproduce them to quadrature-convergence
tolerance. Any refactor, port or node-rule change that shifts a board beyond
these tolerances is a regression, not noise: there is no Monte Carlo here.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   abilities_from_block_race)
from winning.factor.polish import polish_race, race_jacobian
from winning.factor.races import race_probabilities

G = np.load(Path(__file__).parent / "golden" / "race_boards.npz")


def _tv(a, b):
    return 0.5 * np.abs(a - b).sum()


# ------------------------------------------------------------ golden files
def test_golden_block():
    p = block_race_probabilities(G["mu"], G["cl"], G["v"], G["D"])
    assert _tv(p, G["block"]) < 3e-4


def test_golden_nested():
    p = nested_race_probabilities(G["mu"], G["cl"], G["v"], G["D"],
                                  coupling=G["g"], gamma=1.0)
    assert _tv(p, G["nested"]) < 5e-4


def test_golden_rank2_blocks():
    p = block_race_probabilities(G["mu"], G["cl"], G["V2"], G["D"], qa=11)
    assert _tv(p, G["rank2"]) < 5e-4


def test_golden_tree():
    p = tree_race_probabilities(G["mu"], G["cl"], G["v"], G["D"],
                                G["parent"], G["lam"], points=513)
    assert _tv(p, G["tree"]) < 1e-3


# ------------------------------------------------------- convergence order
def test_lattice_convergence_is_monotone_and_fast():
    errs = [_tv(block_race_probabilities(G["mu"], G["cl"], G["v"], G["D"],
                                         points=pts), G["block"])
            for pts in (65, 129, 257)]
    assert errs[1] < errs[0] and errs[2] <= errs[1] * 1.5
    assert errs[2] < 3e-4


def test_node_convergence_is_monotone():
    errs = [_tv(block_race_probabilities(G["mu"], G["cl"], G["v"], G["D"],
                                         qa=q, points=1025), G["block"])
            for q in (3, 5, 9)]
    assert errs[1] < errs[0] and errs[2] < 5e-4


# ------------------------------------------------- statistically rigorous MC
def test_block_race_against_mc_with_zscores():
    mu, cl, v, D = G["mu"], G["cl"], G["v"], G["D"]
    p = block_race_probabilities(mu, cl, v, D, points=1025, qa=15)
    rng = np.random.default_rng(7)
    n, C, M = len(mu), cl.max() + 1, 400_000
    cnt = np.zeros(n)
    for a in range(0, M, 50_000):
        m = min(50_000, M - a)
        A = rng.standard_normal((m, C))
        Y = -mu + v * A[:, cl] + np.sqrt(D) * rng.standard_normal((m, n))
        np.add.at(cnt, np.argmax(Y, axis=1), 1.0)
    mc = cnt / M
    se = np.sqrt(np.maximum(p * (1 - p), 1e-12) / M)
    z = (mc - p) / se
    assert np.abs(z).max() < 5.0
    # chi-square-style global check: sum z^2 ~ chi2(n)
    assert (z ** 2).sum() < stats.chi2.ppf(0.999, df=n)


# ---------------------------------------------------------------- KKT etc.
def test_polish_kkt_stationarity_and_idempotence():
    rng = np.random.default_rng(5)
    n = 12
    mu = rng.normal(0, 0.8, n); mu -= mu.mean()
    g = rng.normal(0, 0.6, n); g -= g.mean(); V = g[:, None]
    D = 0.5 + 0.5 * rng.random(n)
    p0 = race_probabilities(mu, V=V, D=D)
    big = int(np.argmax(p0))
    sector = [int(i) for i in np.argsort(-p0)[:4]]
    caps = np.full(n, np.nan); caps[big] = 0.8 * p0[big]
    p1, mu1, info = polish_race(mu0=mu, V=V, D=D, name_caps=caps,
                                groups=[(sector, 0.9 * p0[sector].sum())])
    assert info["max_violation"] < 1e-7
    # KKT stationarity: (mu1 - mu) must lie in the span of the active
    # constraints' gradients (rows of -A_act J), with nonnegative multipliers
    from winning.factor.polish import concentration_matrix
    A, b = concentration_matrix(n, name_caps=caps,
                                groups=[(sector, 0.9 * p0[sector].sum())])
    J = race_jacobian(mu1, V=V, D=D)
    slack = b - A @ p1
    act = slack < 1e-5
    assert act.any()
    Gm = -(A[act] @ J)                       # gradients of the active b - A p
    # proper KKT check: NONNEGATIVE least squares for the multipliers (a free
    # lstsq can split overlapping gradients with a spurious negative)
    from scipy.optimize import nnls
    lam, resid = nnls(Gm.T, (mu1 - mu))
    denom = max(np.linalg.norm(mu1 - mu), 1e-12)
    assert resid / denom < 5e-2
    # idempotence: polishing the polished race is a no-op
    p2, mu2, info2 = polish_race(mu0=mu1, V=V, D=D, name_caps=caps,
                                 groups=[(sector, 0.9 * p0[sector].sum())])
    assert np.abs(mu2 - mu1).max() < 1e-4


def test_inversion_bounds_semantics_for_subresolution_targets():
    rng = np.random.default_rng(4)
    mu, cl, v, D = G["mu"], G["cl"], G["v"], G["D"]
    p = block_race_probabilities(mu, cl, v, D).copy()
    p[3] = 1e-30                              # sub-resolution "measurement"
    p = p / p.sum()
    mu_hat, resid, _ = abilities_from_block_race(p, cl, v, D)
    assert np.isfinite(mu_hat).all()
    # entry 3 gets a finite bound, not a runaway; the rest still calibrate
    others = np.ones(len(p), bool); others[3] = False
    p_back = block_race_probabilities(mu_hat, cl, v, D)
    assert 0.5 * np.abs(p_back[others] / p_back[others].sum()
                        - p[others] / p[others].sum()).sum() < 5e-3
