"""A kink-split piece above KINK_MAX_ORDER is split into equal panels (#835).

One Gauss rule per piece is O(m^2) to build in the browser (leggauss is
O(m^3) here), and a large points= gave one piece thousands of nodes: a
32,769-point Laplace inverse blocked for seconds before its first sweep.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from winning.factor.races import KINK_MAX_ORDER, abilities_from_race, kink_lattice

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
KINKS = [0.0]


def _lattice(L):
    x = np.linspace(-12.0, 12.0, L)
    return x, np.array([-0.4, 0.1, 0.5]), np.array([1.0, 0.7, 1.3])


@pytest.mark.parametrize("L", [257, 513, 4097, 32769])
def test_panels_keep_the_budget_and_integrate_exactly(L):
    x, M, sd = _lattice(L)
    xs, ws = kink_lattice(x, M, sd, KINKS)
    assert ws.sum() == pytest.approx(x[-1] - x[0], rel=1e-13)
    assert np.all(np.diff(np.sort(xs)) > 0)
    # a cubic is integrated exactly by every Gauss panel
    assert np.sum(ws * xs ** 3) == pytest.approx((x[-1] ** 4 - x[0] ** 4) / 4, abs=1e-8 * x[-1] ** 4)
    assert len(xs) <= L + 8 * KINK_MAX_ORDER


def test_a_large_lattice_inverse_is_fast_and_agrees():
    t = time.time()
    mu_big = abilities_from_race(np.array([0.2, 0.3, 0.5]), base="laplace", points=32769, n_iter=60)
    assert time.time() - t < 20                         # was minutes (leggauss of order ~16k)
    mu_mid = abilities_from_race(np.array([0.2, 0.3, 0.5]), base="laplace", points=4097, n_iter=60)
    assert np.abs(mu_big - mu_mid).max() < 1e-3


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("L", [513, 4097])
def test_browser_builds_the_same_lattice(L):
    x, M, sd = _lattice(L)
    xs, ws = kink_lattice(x, M, sd, KINKS)
    script = f"""
import {{ kinkLattice }} from './docs/js/winning/core.mjs';
const L = {L}, x = Array.from({{ length: L }}, (_, i) => -12 + 24 * i / (L - 1));
const r = kinkLattice(x, {json.dumps(M.tolist())}, {json.dumps(sd.tolist())}, {json.dumps(KINKS)});
console.log(JSON.stringify([r.xs, r.ws]));
"""
    jx, jw = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout)
    assert len(jx) == len(xs)
    np.testing.assert_allclose(np.sort(jx), np.sort(xs), atol=1e-12)
    assert abs(sum(jw) - ws.sum()) < 1e-12
