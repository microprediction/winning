"""An Aitken trial fires only while the residual contracts as the step does (#956).

On this exact self-generated target the plain sweeps sit on a plateau for
~45 sweeps and then finish superlinearly (9.3, 0.13, 2e-5, 1e-10). Trials
fired in that finish at residual/step ratios 0.60/0.86 and 0.63/0.88,
each raised the residual 2-9x, and the inverse ended 13.6 off at the
default 60-sweep budget where the plain sweeps converge.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.factor.races import race_probabilities, abilities_from_race

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
D = [41.762407878182444, 0.09543068227692414, 0.014468331935236405, 0.30750102862954337,
     5.76767363738077, 39.60275014004311, 0.05313878565886935, 5.374508765776025,
     1.243508995177201, 0.02860465678144087]
TRUTH = [-3.2047748303040864, -2.7799773643724626, -1.1310159973800182, -0.825104952324182,
         -0.40134965870529415, 0.3350836188532412, 0.16529664136469366, 0.2920818901620805,
         0.8125011758878828, 2.418152273911983]


@pytest.mark.parametrize("points", [513, 1001, 1025])
def test_exact_target_converges_within_the_default_budget(points):
    target = race_probabilities(np.array(TRUTH), D=np.array(D), points=points)
    mu, info = abilities_from_race(target, D=np.array(D), points=points, n_iter=60,
                                   tol=1e-8, return_info=True)
    back = race_probabilities(mu, D=np.array(D), points=points)
    assert info["converged"], info                     # was 13.6 of log residual
    assert np.abs(np.log(back) - np.log(target)).max() < 1e-8


def test_gate_constant_is_the_documented_one():
    from winning.factor.races import AITKEN_AGREE
    assert AITKEN_AGREE == 0.15


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_exact_target_converges_within_the_default_budget():
    script = f"""
import './docs/js/winning/structures.mjs';
import {{ raceProbabilities, abilitiesFromRace }} from './docs/js/winning/races.mjs';
console.warn = () => {{}};
const D = {json.dumps(D)}, truth = {json.dumps(TRUTH)};
console.log(JSON.stringify([513, 1001, 1025].map(points => {{
  const target = raceProbabilities(truth, {{ D, points }});
  const fit = abilitiesFromRace(target, {{ D, points, nIter: 60, tol: 1e-8, returnInfo: true }});
  const back = raceProbabilities(fit.mu, {{ D, points }});
  return {{ converged: fit.converged,
           residual: Math.max(...back.map((p, i) => Math.abs(Math.log(p) - Math.log(target[i])))) }};
}})));
"""
    got = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout)
    for r in got:
        assert r["converged"] and r["residual"] < 1e-8, got
