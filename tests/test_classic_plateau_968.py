"""A flat miss with moving offsets is not a plateau (#968).

The middle runner of this book is priced at exactly zero for a few steps
while its offset travels toward positive mass, so the maximum miss is
pinned at its 0.0018 target. The #923 rule took three such iterates for a
plateau and stopped 1.8e-3 off; the iteration reaches 1e-9. A stall now
also needs the update to have left the offsets where they were, in
python, the browser and R.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from winning.classic.lattice import state_prices_from_offsets
from winning.classic.lattice_calibration import solve_for_implied_offsets

NODE = shutil.which("node")
D = np.zeros(49)
D[[15, 26, 35]] = [0.1, 0.6, 0.3]
TRUTH = [2.5, 15.8, -4.8]


@pytest.fixture
def numpy_path(monkeypatch):
    import winning.classic.lattice_calibration as lc
    monkeypatch.setattr(lc, "_HAVE_RUST", False)


def test_a_temporary_zero_price_does_not_stop_the_inverse(numpy_path):
    target = np.asarray(state_prices_from_offsets(D, TRUTH))
    got = solve_for_implied_offsets(target, D, nIter=100)
    miss = float(np.max(np.abs(np.asarray(state_prices_from_offsets(D, got)) - target)))
    assert miss < 1e-8, miss                    # was 1.8e-3


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_temporary_zero_price_does_not_stop_the_inverse():
    script = f"""
import {{ solveForImpliedOffsets, statePricesFromOffsets }} from "./docs/js/winning/classic.mjs";
console.warn = () => {{}};
const d = {json.dumps(D.tolist())};
const target = statePricesFromOffsets(d, {json.dumps(TRUTH)});
const mu = solveForImpliedOffsets(target, d, {{ nIter: 100 }});
const back = statePricesFromOffsets(d, mu);
console.log(JSON.stringify(Math.max(...back.map((p, i) => Math.abs(p - target[i])))));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                         text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) < 1e-8
