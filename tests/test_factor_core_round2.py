"""Regressions for the second factor-core issue batch. Each test names its
issue; each was run against the pre-fix tree and failed there."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.factor.topk import abilities_from_topk, top_k_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    return json.loads(out.stdout)


def _mod(name):
    # a file:// URI, JSON-quoted, so Windows paths are legal specifiers
    return json.dumps(Path(ROOT, "docs", "js", "winning", name).as_uri())


# ---------------------------------------------------------------- #583

MU_583 = np.array([-2.4085533796267504, -2.371749066692294,
                   0.34486967818371195, -1.761616714882222,
                   4.096640356222963, 2.100409126794591])
SD_583 = np.array([0.3622773278819501, 1.9392652513644468,
                   1.1350172886349525, 0.8731304646755176,
                   2.865406646646641, 0.6217100227308292])


def test_aitken_jump_is_a_checked_trial_583():
    """An exact top-1 target: the unchecked 539x extrapolation ended at
    logit residual 715 with abilities off by ~490."""
    D = SD_583 ** 2
    q = top_k_probabilities(MU_583, 1, D=D, points=257)
    fit, info = abilities_from_topk(q, 1, D=D, points=257, n_iter=120,
                                    tol=1e-8, return_info=True)
    err = fit - MU_583
    assert np.abs(err - err.mean()).max() < 1e-6
    assert info["max_logit_residual"] < 1e-5          # was 715
    rep = top_k_probabilities(fit, 1, D=D, points=257)
    assert np.abs(rep - q).max() < 1e-9


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_aitken_jump_is_a_checked_trial_583():
    got = _node(f"""
import {{ topKProbabilities, abilitiesFromTopk }} from {_mod("topk.mjs")};
console.warn = () => {{}};
const mu = {json.dumps(MU_583.tolist())};
const D = {json.dumps((SD_583 ** 2).tolist())};
const q = topKProbabilities(mu, 1, {{D, points: 257}});
const fit = abilitiesFromTopk(q, 1, {{D, points: 257, nIter: 120, tol: 1e-8,
                                     returnInfo: true}});
const rep = topKProbabilities(fit.mu, 1, {{D, points: 257}});
console.log(JSON.stringify({{mu: fit.mu, info: fit.info, rep, q}}));
""")
    err = np.array(got["mu"]) - MU_583
    assert np.abs(err - err.mean()).max() < 1e-6       # was ~900
    assert got["info"]["maxLogitResidual"] < 1e-5      # was 715
    assert np.abs(np.array(got["rep"]) - got["q"]).max() < 1e-9


# ---------------------------------------------------------------- #524

@pytest.mark.parametrize("d0", [0.003, 0.01, 0.03, 0.1])
def test_core_inverse_hazard_is_analytic_524(d0):
    """A narrow runner's density underflowed, was floored at 1e-300 while
    log S kept falling, and inf - inf made every ability NaN."""
    from winning.factor.core import (abilities_from_probabilities_factor,
                                     win_probabilities_factor)
    D = np.array([d0, 1.0, 1.0])
    V, F, W = np.zeros((3, 1)), np.zeros((1, 1)), np.ones(1)
    truth = np.array([0.5, 0.0, -0.5])
    target = win_probabilities_factor(truth, V, D, F, W)
    with np.errstate(over="raise", invalid="raise"):
        mu, info = abilities_from_probabilities_factor(
            target, V, D, F, W, return_info=True)
    assert np.isfinite(mu).all() and info["converged"]
    rep = win_probabilities_factor(mu, V, D, F, W)
    assert np.abs(rep - target).max() < 1e-6


# ---------------------------------------------------------------- #510

def _hier_510():
    from winning.factor.structures import Blocks, Nested, Tree
    c = np.array([0, 0, 1, 1])
    L = np.array([0.2, 0.3, -0.1, 0.4])
    g = np.array([0.1, -0.2, 0.3, -0.1])
    return [lambda D: Blocks(c, L, D),
            lambda D: Nested(c, L, D, g, 0.5),
            lambda D: Tree(c, L, D, [2, 2, -1], [0.3, 0.3, 0.2])]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_structured_inverse_reads_a_scalar_D_510(which):
    """structure_variances took len() of a 0-d D: the forward priced a
    scalar-D Blocks/Nested race and its own inverse raised TypeError."""
    from winning.factor.races import abilities_from_race, race_probabilities
    from winning.factor.structures import structure_variances
    make = _hier_510()[which]
    mu = np.array([-0.5, -0.1, 0.2, 0.4])
    mu -= mu.mean()
    s, v = make(1.0), make(np.ones(4))
    assert s.n == 4
    assert np.array_equal(structure_variances(s), structure_variances(v))
    p = race_probabilities(mu, structure=s, points=257)
    m, info = abilities_from_race(p, structure=s, points=257,
                                  return_info=True)
    mv = abilities_from_race(p, structure=v, points=257)
    assert info["converged"] and np.abs(m - mu).max() < 1e-7
    assert np.abs(m - mv).max() < 1e-12
