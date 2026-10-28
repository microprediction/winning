"""Browser skewNormalDensity half-width validation (#868). Each
regression below was run against the pre-fix tree, where it failed.

L went straight into `2 * L + 1`: 2.5, true and "3" came back as length
6, 3 (mass 0.33) and 7 vectors, and 2 as a length-5 vector every classic
consumer refuses. The python constructor already asserts an int L."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
CLASSIC = json.dumps((ROOT / "docs" / "js" / "winning" / "classic.mjs").as_uri())

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


def test_browser_refuses_unusable_half_widths_868():
    got = _node(f"""
import {{ skewNormalDensity }} from {CLASSIC};
const out = [];
for (const L of [2.5, 2, 0, -3, true, "3", NaN, Infinity, -Infinity, null, undefined, [3]]) {{
  try {{ out.push(["ok", skewNormalDensity(L, 0.1, {{ a: 0 }}).length]); }}
  catch (e) {{ out.push(["err", e.message]); }}
}}
console.log(JSON.stringify(out));
""")
    for kind, msg in got:
        assert kind == "err", msg
        assert msg.startswith("skewNormalDensity needs"), msg


def test_browser_whole_half_widths_unchanged_868():
    got = _node(f"""
import {{ skewNormalDensity, statePricesFromOffsets }} from {CLASSIC};
const d3 = skewNormalDensity(3, 0.1, {{ a: 0 }});
console.log(JSON.stringify([d3, statePricesFromOffsets(d3, [0, 1]),
                            skewNormalDensity(50, 0.1), skewNormalDensity(15, 0.2, {{ a: 1.5 }})]));
""")
    d3, p, d50, d15 = got
    assert len(d3) == 7 and abs(sum(d3) - 1) < 1e-12 and min(d3) >= 0
    assert len(p) == 2 and abs(sum(p) - 1) < 1e-12
    # the browser constructor still matches python's
    assert np.allclose(d50, skew_normal_density(50, 0.1), rtol=0, atol=1e-12)
    assert np.allclose(d15, skew_normal_density(15, 0.2, a=1.5), rtol=0, atol=1e-12)
