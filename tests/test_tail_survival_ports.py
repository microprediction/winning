"""Stable normal survival and base-aware bulk windows across ports (#96, #106).

`1 - ndtr(z)` cancels to exactly zero past about 8.3 sd, so a 20-sd
longshot came out ~95x too unlikely in pure python, the browser and R
(Rust already used log_ndtr(-z)). And the bulk window hard-coded the
normal survival for every base in R and the browser, truncating a
custom polynomial tail at any point count. R's side is pinned in
r/winning/tests/testthat/test-races.R; this file pins python and drives
the browser modules through node.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from winning.factor.core import win_probabilities
from winning.factor.races import race_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")

# direct stable quadrature of the middle runner of mu=[0,20,40], D=1
MIDDLE = 1.044243791881266e-45
# scipy quad of f_i prod_{j!=i} S_j for the custom Student-t(3) fixture
T3_REF = [0.5662010847332788, 0.3610182079892741, 0.0727807072774470]


def test_python_normal_survival_holds_a_20_sd_longshot():
    p = race_probabilities(np.array([0.0, 20.0, 40.0]), D=np.ones(3),
                           window="span", points=16385)
    assert abs(p[1] / MIDDLE - 1) < 1e-8          # was 1.1e-47 (93x low)
    q = win_probabilities(np.array([0.0, 20.0, 40.0]),
                          x=np.linspace(-12, 60, 20001))
    assert abs(q[1] / MIDDLE - 1) < 1e-8


_JS = r"""
import { raceProbabilities } from "%s";
const t3 = z => {
  const f = 2 / (Math.PI * (1 + z*z)**2);
  const F = 0.5 + (Math.atan(z) + z/(1 + z*z)) / Math.PI;
  const fp = -8*z / (Math.PI * (1 + z*z)**3);
  return [Math.max(1 - F, 1e-300), f, fp];
};
console.warn = () => {};
const tail = raceProbabilities([0, 20, 40], {D: [1, 1, 1], points: 16385,
                                             window: "span"});
const t = raceProbabilities([0, 1, 2], {D: [64, 1, 1], base: t3,
                                        points: 32001, window: "bulk"});
console.log(JSON.stringify({tail, t}));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_survival_and_base_aware_bulk_window():
    races = os.path.join(ROOT, "docs", "js", "winning", "races.mjs")
    out = subprocess.run([NODE, "--input-type=module", "-e", _JS % races],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    got = json.loads(out.stdout)
    assert abs(got["tail"][1] / MIDDLE - 1) < 1e-8     # was 92x low
    # was 1.2245e-4 off on the first share at any point count
    assert max(abs(a - b) for a, b in zip(got["t"], T3_REF)) < 1e-7
