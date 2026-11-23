"""The classic inverse returns offsets the public forward prices as it did (#1046).

The iteration prices each offset on its own, and _low_high pins anything
past +/-(L-2) to the edge, so an iterate could drift past the edge with
no change in its own reprice. The public forward translates such a book
back in range first (_edge_centred), which pinned the other runners
together at the opposite edge: a calibration the iteration scored at
3.8e-4 repriced 9.4e-2 off. The returned offsets are now pinned to
[-(L-2), L-2], the offsets the iteration actually priced.
"""
import json
import shutil
import subprocess
import warnings

import numpy as np
import pytest

import winning.classic.lattice_calibration as lc
from winning.classic.lattice import (implied_L, state_prices_from_offsets, _exact_offset_cdfs,
                                     exact_state_prices_from_cdfs)
from winning.classic.lattice_calibration import solve_for_implied_offsets

ATOMS = np.zeros(15)
ATOMS[0], ATOMS[1], ATOMS[9] = 0.5292435891016793, 0.3029411994181423, 0.16781521148017847
TRUTH = [-3.067361030359437, -2.884203651711936, 4.646685322840478, -2.6360814331198537]
TARGET = [float(x) for x in state_prices_from_offsets(ATOMS, TRUTH)]
NODE = shutil.which("node")


def _miss(density, target, mu):
    return float(np.max(np.abs(np.asarray(state_prices_from_offsets(density, mu)) - np.asarray(target))))


@pytest.fixture(params=["default", "numpy"])
def backend(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setattr(lc, "_HAVE_RUST", False)


def test_offsets_stay_on_the_lattice_and_the_public_forward_agrees(backend):
    L = implied_L(ATOMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # warned 9.37e-2
        mu = np.asarray(solve_for_implied_offsets(TARGET, ATOMS))
    assert np.all(np.abs(mu) <= L - 2)                         # was 8.26 at L=7
    internal = exact_state_prices_from_cdfs(_exact_offset_cdfs(ATOMS, mu)[1])
    np.testing.assert_allclose(state_prices_from_offsets(ATOMS, mu), internal, rtol=0, atol=1e-12)
    assert _miss(ATOMS, TARGET, mu) < 1e-3                     # was 9.4e-2


def test_in_range_results_are_unchanged():
    # pinning leaves the iteration alone: an in-range calibration is the iterate itself
    d = np.zeros(31)
    d[[2, 18, 23, 26, 28, 30]] = [0.10, 0.05, 0.30, 0.05, 0.45, 0.05]
    target = state_prices_from_offsets(d, [5.5, 1.0, -7.0, 6.0])
    mu = solve_for_implied_offsets(target, d)
    assert np.all(np.abs(mu) < implied_L(d) - 2)
    assert _miss(d, target, mu) <= 1e-6


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_port_stays_on_the_lattice():
    script = f"""
import {{ solveForImpliedOffsets }} from "./docs/js/winning/classic.mjs";
console.log(JSON.stringify(solveForImpliedOffsets({json.dumps(TARGET)}, {json.dumps(list(map(float, ATOMS)))})));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    mu = json.loads(out.stdout.strip().splitlines()[-1])
    assert max(abs(m) for m in mu) <= implied_L(ATOMS) - 2
    assert _miss(ATOMS, TARGET, mu) < 1e-3
