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
