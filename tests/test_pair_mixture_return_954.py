"""Certify the evaluated pair-mixture point rather than its next proposal (#954)."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]


def fixture(scale, reverse):
    order = [1, 0] if reverse else [0, 1]
    return (np.array([0.012575, -0.012575])[order] * scale,
            dict(V=(np.array([[1.0], [0.0]])[order] * scale).tolist(),
                 D=(np.array([8.7e-6, 1.8e-6])[order] * scale**2).tolist(),
                 F=[[-0.5], [0.0], [4.0]], W=[0.37, 0.088, 0.02]))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tol", [1e-8, 1e-12])
def test_evaluated_point_is_certified_by_public_forward(scale, reverse, tol):
    truth, opts = fixture(scale, reverse)
    target = race_probabilities(truth, **opts)
    mu, info = abilities_from_race(target, **opts, tol=tol, return_info=True)
    assert info["converged"] and info["iterations"] == 0, info
    assert np.isfinite(mu).all() and mu.sum() == 0.0
    back = race_probabilities(mu, **opts)
    residual = float(np.max(np.abs(np.log(back) - np.log(target))))
    assert residual < tol
    assert info["max_log_residual"] == pytest.approx(residual, abs=1e-15)
    # The law has no negligible nodes: this does not conceal #941 pruning.
    assert min(opts["W"]) / sum(opts["W"]) > 0.04


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_evaluated_points_and_tighter_tolerance():
    cases = [dict(truth=fixture(scale, reverse)[0].tolist(),
                  opts=fixture(scale, reverse)[1], tol=tol)
             for scale in (1e-3, 1.0, 1e3)
             for reverse in (False, True) for tol in (1e-8, 1e-12)]
    script = f'''
import {{raceProbabilities, abilitiesFromRace}} from './docs/js/winning/races.mjs';
const cases = {json.dumps(cases)};
const results = cases.map(({{truth, opts, tol}}) => {{
  const target = raceProbabilities(truth, opts);
  const fit = abilitiesFromRace(target, {{...opts, tol, returnInfo:true}});
  const back = raceProbabilities(fit.mu, opts);
  return {{fit, residual:Math.max(...back.map((p,i)=>Math.abs(Math.log(p)-Math.log(target[i]))))}};
}});
console.log(JSON.stringify(results));
'''
    got = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script],
                                    cwd=ROOT, capture_output=True, text=True,
                                    check=True).stdout)
    for case, result in zip(cases, got):
        fit = result["fit"]
        assert fit["converged"] and fit["iterations"] == 0, result
        assert np.isfinite(fit["mu"]).all() and sum(fit["mu"]) == 0.0
        assert result["residual"] < case["tol"]
        assert fit["maxLogResidual"] == pytest.approx(result["residual"], abs=1e-15)
