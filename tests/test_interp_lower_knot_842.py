"""np.interp at a duplicated lower knot, browser port (#842). Each
regression below was run against the pre-fix tree, where it failed.

interpClamped returned fp[0] for x <= xp[0], so at a duplicated lower
knot it took the FIRST of the plateau where numpy takes the last. In the
classic inverse a runner priced at exactly zero mid-calibration then read
its offset off the wrong end of the zero-price plateau: a self-generated
five-runner book on a compact law missed by 0.0152 with a false warning,
where python's reference calibration reaches 2.1e-6. The same rule in
Rust interp1 and R np_interp is tested in those ports."""
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
JS = ROOT / "docs" / "js" / "winning"
CORE = json.dumps((JS / "core.mjs").as_uri())
CLASSIC = json.dumps((JS / "classic.mjs").as_uri())

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DENSITY = [0] * 9 + [0.0625, 0.125, 0.1875, 0.25, 0.1875, 0.125, 0.0625] + [0] * 9
TRUTH = [-0.3870526359606288, -3.8859224668266945, -0.5618939753478802,
         -6.990429141603045, 0.6690584211927497]

CASES = [
    (0, [0, 0, 0.5], [10, 5, 0]),               # duplicated lower knot: the fix
    (0, [0, 0, 0, 0.5], [10, 5, 3, 0]),
    (-1, [0, 0, 0.5], [10, 5, 0]),              # below the lower knot
    (1, [0, 1, 1, 2], [0, 5, 7, 9]),            # duplicated interior knot
    (2, [0, 2, 2], [0, 5, 7]),                  # duplicated upper knot
    (0, [0, 1, 2], [4, 5, 7]),                  # ordinary ascending knots
    (0.5, [0, 1, 2], [4, 5, 7]),
    (1, [0, 1, 2], [4, 5, 7]),
    (3, [0, 1, 2], [4, 5, 7]),
]


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


def test_browser_interp_matches_numpy_at_every_knot_842():
    got = _node(f"""
import {{ interpClamped }} from {CORE};
console.log(JSON.stringify({json.dumps(CASES)}.map(([x, xp, fp]) => interpClamped(x, xp, fp))));
""")
    for (x, xp, fp), g in zip(CASES, got):
        assert g == np.interp(x, xp, fp), (x, xp, fp, g)


def test_browser_compact_inverse_follows_python_842():
    got = _node(f"""
import {{ statePricesFromOffsets as f, solveForImpliedOffsets as inv }} from {CLASSIC};
const warned = [];
console.warn = (...a) => warned.push(String(a[0]));
const d = {json.dumps(DENSITY)};
const target = f(d, {json.dumps(TRUTH)});
const back = f(d, inv(target, d));
console.log(JSON.stringify({{ target, back, warned }}));
""")
    err = max(abs(a - b) for a, b in zip(got["target"], got["back"]))
    assert err < 1e-5, err                      # was 0.0152
    assert not got["warned"]
    from winning.classic.lattice import state_prices_from_offsets
    from winning.classic.lattice_calibration import solve_for_implied_offsets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = state_prices_from_offsets(DENSITY, solve_for_implied_offsets(got["target"], DENSITY))
    assert np.allclose(got["back"], py, rtol=0, atol=1e-9)
