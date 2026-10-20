"""Classic extended forward, the +-(L - 2) endpoint (#773). Each
regression was run against the pre-fix tree, where it failed.

The browser (and R) hanging test for a support narrower than the lattice
marked an offset of exactly +-(L - 2) as hanging, though low_high
represents it exactly; [0, 1] at L = 3 was dilated and clustered to
[0.634, 0.366] against the exact [7/9, 2/9]. Python's bound was already
strict."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.classic.lattice import (state_prices_from_extended_offsets,
                                     state_prices_from_offsets)

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
CLASSIC = json.dumps((ROOT / "docs" / "js" / "winning" / "classic.mjs").as_uri())

L3 = [0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0]          # L = 3, support {-1, 0, 1}
EXACT = [7 / 9, 2 / 9]


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


def test_python_prices_the_l_minus_2_endpoint_exactly_773():
    # the reference the browser now agrees with
    for off, want in [([0, 1], EXACT), ([0, -1], EXACT[::-1])]:
        assert np.allclose(state_prices_from_extended_offsets(L3, off), want,
                           rtol=0, atol=1e-15)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_prices_the_l_minus_2_endpoint_exactly_773():
    got = _node(f"""
import {{ abilityImpliedStatePrices as f, statePricesFromOffsets as g }} from {CLASSIC};
const d = {json.dumps(L3)};
const out = {{}};
for (const off of [[0, 1], [0, -1], [1, 0], [-1, 0]])
  out[JSON.stringify(off)] = [0, 1, 2, 3].map(md => f(off, d, {{ maxDepth: md }})).concat([g(d, off)]);
console.log(JSON.stringify(out));
""")
    for key, rows in got.items():
        direct = rows[-1]
        for p in rows[:-1]:                    # every maxDepth, was 0.634 / [1, 0]
            assert np.allclose(p, direct, rtol=0, atol=1e-15), (key, p)
        want = EXACT if json.loads(key)[0] < json.loads(key)[1] else EXACT[::-1]
        assert np.allclose(direct, want, rtol=0, atol=1e-15)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_endpoint_neighbours_unchanged_773():
    # values pinned from origin/main 7af3ed1c: just inside the endpoint was
    # already exact, just past it still hangs and clusters, and a W >= L
    # field is untouched
    got = _node(f"""
import {{ abilityImpliedStatePrices as f, skewNormalDensity }} from {CLASSIC};
const d = {json.dumps(L3)};
console.log(JSON.stringify([f([0, 1 - 1e-9], d), f([0, 1 + 1e-9], d),
  f([0, 0.7, 1.3, 2.1], skewNormalDensity(200, 0.1, {{ a: 0 }}), {{ unit: 0.1 }}),
  f([-0.5, 0, 0.3, 2.1, 6], skewNormalDensity(50, 0.1, {{ a: 1 }}), {{ unit: 0.1 }})]));
""")
    pinned = [[0.7777777775000001, 0.22222222250000007],
              [0.633744856100823, 0.36625514389917707],
              [0.6049811058833535, 0.2601479044907512, 0.1086783458341974,
               0.026192643791697244],
              [0.5672982284762254, 0.27027574567228596, 0.16087563506704702,
               0.001550390784063666, 3.7779460993255496e-13]]
    for p, want in zip(got, pinned):
        assert np.allclose(p, want, rtol=1e-15, atol=1e-15), (p, want)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_wider_narrow_support_endpoint_773():
    # L = 5, support {-2..2}: the runner at exactly -(L - 2) = -3 keeps its
    # whole support on the lattice, so the field is the exact primitive
    d = [0.0] * 11
    for k in range(3, 8):
        d[k] = 0.2
    off = [-1, 0, 1, 1, -1, 0, -3, 0]
    got = _node(f"""
import {{ abilityImpliedStatePrices as f }} from {CLASSIC};
console.log(JSON.stringify(f({json.dumps(off)}, {json.dumps(d)})));
""")
    want = state_prices_from_offsets(d, [float(o) for o in off])
    assert np.allclose(got, want, rtol=0, atol=1e-15)    # was 0.21 off
