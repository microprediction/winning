"""Classic extended forward, the no-chance removal test (#794). Each
regression below was run against the pre-fix tree, where it failed.

The test compared a runner's raw fractional gap with the integer support
width W, but low_high mixes a fractional offset's floor and ceil shifts,
so its support reaches one cell further. [0, 1.0001] on a two-atom law
(W = 1) was priced [1, 0] against the exact [0.8750125, 0.1249875].
Python also tested any() over the bad indices, which is falsy when the
only bad runner is index 0, so its answer depended on field order."""
import itertools
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

L = 15
TWO_ATOM = [0.0] * (2 * L + 1)
TWO_ATOM[L] = TWO_ATOM[L + 1] = 0.5             # support width W = 1
EXACT = [0.8750125, 0.1249875]                  # 0.9999 * 0.125 dead-heat claim

# gaps around W = 1: just below, equal, just above, mid, top of the
# (W, W + 1) window where the fractional runner still ties
VIABLE = [0.9999, 1.0, 1.0001, 1.5, 1.9999]
# integer or fractional gaps whose supports are strictly disjoint: these
# still take the zero-price removal path, as before
DISJOINT = [2.0, 2.5, 3.0, 7.25]


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


def _fields():
    for g in VIABLE + DISJOINT:
        yield [0.0, g]
        yield [g, 0.0]
        yield [0.5, 0.5 + g]                    # a fractional best as well


def test_python_extended_matches_exact_for_fractional_gaps_794():
    for off in _fields():
        got = state_prices_from_extended_offsets(TWO_ATOM, off)
        want = state_prices_from_offsets(TWO_ATOM, off)
        assert np.allclose(got, want, rtol=0, atol=1e-15), (off, got, want)
    for off, want in [([0, 1.0001], EXACT), ([1.0001, 0], EXACT[::-1])]:
        assert np.allclose(state_prices_from_extended_offsets(TWO_ATOM, off),
                           want, rtol=0, atol=1e-15)


def test_python_disjoint_runners_still_priced_zero_794():
    for g in DISJOINT:
        assert list(state_prices_from_extended_offsets(TWO_ATOM, [0.0, g])) == [1.0, 0]
        assert list(state_prices_from_extended_offsets(TWO_ATOM, [g, 0.0])) == [0, 1.0]


def test_python_removal_is_permutation_equivariant_794():
    # a lone bad runner at index 0 used to skip the removal branch
    base = [30.0, 0.0, 0.3, 1.0]
    want = np.array(state_prices_from_extended_offsets(TWO_ATOM, base[1:] + base[:1]))
    want = np.concatenate([want[-1:], want[:-1]])
    assert want[0] == 0
    for perm in itertools.permutations(range(4)):
        off = [base[i] for i in perm]
        got = state_prices_from_extended_offsets(TWO_ATOM, off)
        assert np.allclose(got, want[list(perm)], rtol=0, atol=1e-15), (off, got)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_extended_matches_exact_for_fractional_gaps_794():
    fields = list(_fields())
    got = _node(f"""
import {{ abilityImpliedStatePrices as f, statePricesFromOffsets as g }} from {CLASSIC};
const d = {json.dumps(TWO_ATOM)};
console.log(JSON.stringify({json.dumps(fields)}.map(o => [f(o, d), g(d, o)])));
""")
    for off, (ext, exact) in zip(fields, got):
        assert np.allclose(ext, exact, rtol=0, atol=1e-15), (off, ext, exact)
        assert np.allclose(ext, state_prices_from_offsets(TWO_ATOM, off), rtol=0, atol=1e-12)
    assert np.allclose(got[fields.index([0.0, 1.0001])][0], EXACT, rtol=0, atol=1e-15)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_ordinary_fields_unchanged_794():
    # integer gaps and a smooth law: the removal test agrees with the old one
    got = _node(f"""
import {{ abilityImpliedStatePrices as f, skewNormalDensity }} from {CLASSIC};
const d = {json.dumps(TWO_ATOM)};
const s = skewNormalDensity(50, 0.1, {{ a: 1 }});
console.log(JSON.stringify([f([0, 1], d), f([0, 2], d), f([3, 0, 1], d),
                            f([-0.5, 0, 0.3, 2.1, 6], s, {{ unit: 0.1 }})]));
""")
    assert np.allclose(got[0], [0.875, 0.125], rtol=0, atol=1e-15)
    assert got[1] == [1, 0]
    assert np.allclose(got[2], [0, 0.875, 0.125], rtol=0, atol=1e-15)
    assert abs(sum(got[3]) - 1) < 1e-12 and got[3][4] < 1e-6
