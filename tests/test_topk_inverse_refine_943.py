"""The top-k inverse refines a failing lattice as its forward does (#943).

The forward doubles a lattice that fails the slot-mass check; the inverse
evaluated one fixed lattice and refused, at its warm start, a target the
public forward had produced with the same 513-point controls.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.factor.topk import abilities_from_topk, top_k_probabilities

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
MU = [-0.10455761128105223, 0.10455761128105223]
D = [0.8967612229487083, 0.021331877150829952]


@pytest.mark.parametrize("points", [513, 1025])
def test_inverse_accepts_its_forward_target(points):
    q = top_k_probabilities(MU, 1, D=D, base="laplace", points=513)
    mu, info = abilities_from_topk(q, 1, D=D, base="laplace", points=points, return_info=True)
    assert info["converged"]
    back = top_k_probabilities(mu, 1, D=D, base="laplace", points=points)
    assert np.abs(back - q).max() < 1e-8


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_inverse_accepts_its_forward_target():
    script = f"""
import {{ topKProbabilities, abilitiesFromTopk }} from './docs/js/winning/topk.mjs';
const mu = {json.dumps(MU)}, D = {json.dumps(D)};
const q = topKProbabilities(mu, 1, {{ D, base: 'laplace', points: 513 }});
console.log(JSON.stringify([513, 1025].map(points => {{
  const f = abilitiesFromTopk(q, 1, {{ D, base: 'laplace', points, returnInfo: true }});
  const back = topKProbabilities(f.mu, 1, {{ D, base: 'laplace', points }});
  return [f.info.converged, Math.max(...back.map((v, i) => Math.abs(v - q[i])))];
}})));
"""
    got = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout)
    for converged, err in got:
        assert converged and err < 1e-8
