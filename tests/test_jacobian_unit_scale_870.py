"""The race Jacobian scales as 1/c at every unit, not only near 1 (#870).

Each log-domain factor of the kernel is capped, so at a unit of 1e-18 a
tied race's off-diagonal came back 0.82 of the exact value in python
(19x it at 1e-20) and 0.47 in the browser, while the forward stayed exact.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest
from pathlib import Path

from winning.factor.polish import race_jacobian

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
EXACT = 1 / (4 * np.sqrt(np.pi))         # s * J[0, 1] for three tied unit normals
SCALES = [1.0, 1e-16, 1e-18, 1e-20, 1e-100, 1e20, 1e100]


@pytest.mark.parametrize("s", SCALES)
def test_tied_race_off_diagonal_is_exact_at_any_unit(s):
    J = race_jacobian(np.zeros(3), D=np.full(3, s * s), points=501)
    assert s * J[0][1] == pytest.approx(EXACT, rel=1e-12)


@pytest.mark.parametrize("c", [1e-19, 1e-60, 1e40])
def test_factor_jacobian_is_equivariant(c):
    V = np.array([[0.3], [-0.2], [0.5]]); mu = np.array([0.1, -0.3, 0.2]); D = np.array([1, 1.5, 0.7])
    J1 = race_jacobian(mu, V=V, D=D, points=501)
    Jc = race_jacobian(mu * c, V=V * c, D=D * c * c, points=501)
    assert np.abs(J1 - c * Jc).max() <= 1e-12 * np.abs(J1).max()


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_tied_race_off_diagonal_is_exact_at_any_unit():
    script = f"""
import './docs/js/winning/structures.mjs';
import {{ raceJacobian }} from './docs/js/winning/polish.mjs';
console.log(JSON.stringify({json.dumps(SCALES)}.map(s => s * raceJacobian([0, 0, 0], {{ D: [s * s, s * s, s * s], points: 501 }})[0][1])));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    for s, v in zip(SCALES, json.loads(out.stdout)):
        assert v == pytest.approx(EXACT, rel=1e-12), s

