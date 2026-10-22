"""The rank-one Gauss-Hermite/midpoint handover is a band, not a switch
(#783).

The default rank-one factor rule switched from 80-node Gauss-Hermite to
an 81-node midpoint-quantile grid at ceil(8 sharp) > 80, and the two
rules disagree there: a 1e-7 change in one runner's sd moved a top-2
membership by 31bp. On (_R1_BLEND_LO, _R1_BLEND_HI] -- the Gauss-Hermite
side of the switch -- the rule is now the weighted union of both rules
with a C1 smoothstep weight; outside it the rule is exactly what it was.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
from scipy.special import ndtri

from winning.factor.core import hermite_nodes
from winning.factor import races
from winning.factor.races import _setup, race_probabilities
from winning.factor.topk import top_k_probabilities
from winning.shapes import as_factor_law

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
# read softly so the regressions below fail on the old code for the right
# reason, not at import
_R1_BLEND_LO = getattr(races, "_R1_BLEND_LO", 9.0)
_R1_BLEND_HI = getattr(races, "_R1_BLEND_HI", 10.0)

A = 10 / np.sqrt(2)
V = np.array([[-A], [A / 3], [A / 3], [A / 3]])   # sharpness 10 / sd0
MU = np.array([-0.7, -0.1, 0.2, 0.8])


def _D(sd0):
    return np.array([sd0 * sd0, 1.0, 1.0, 1.0])


def _rule(sharp):
    _, _, _, F, W, _, _, _ = _setup(MU, V, _D(10.0 / sharp), None, None,
                                    "normal")
    return F, W


def _old_rule(sharp):
    """The pre-#783 dispatch, the rule outside the band must still be."""
    Q = int(np.ceil(8.0 * sharp))
    if Q > 80:
        Q = min(Q, 4001)
        F, W = ndtri((np.arange(Q) + 0.5) / Q)[:, None], np.full(Q, 1.0 / Q)
    else:
        F, W = hermite_nodes(1, Q=int(np.clip(Q, 15, 201)))
    return as_factor_law(F, W, negligible=False)


def test_issue_fixture_is_continuous():
    """The #783 fixture: sd0 = 1 - 1e-7 against sd0 = 1 was a 3.1e-3 jump."""
    lo = top_k_probabilities(MU, 2, V=V, D=_D(1 - 1e-7), points=513)
    mid = top_k_probabilities(MU, 2, V=V, D=_D(1.0), points=513)
    hi = top_k_probabilities(MU, 2, V=V, D=_D(1 + 1e-7), points=513)
    assert np.max(np.abs(lo - mid)) < 1e-7
    assert np.max(np.abs(hi - mid)) < 1e-7


def test_finite_differences_no_longer_diverge():
    """Straddling a jump made the central difference scale like 1/h
    (-1.56e4 at h = 1e-7); across the band it converges."""
    fd = []
    for h in (1e-4, 1e-6, 1e-7):
        pp = top_k_probabilities(MU, 2, V=V, D=_D(1 + h), points=513)
        pm = top_k_probabilities(MU, 2, V=V, D=_D(1 - h), points=513)
        fd.append((pp - pm) / (2 * h))
    assert np.max(np.abs(fd[0] - fd[2])) < 1e-4
    assert np.max(np.abs(fd[2])) < 1.0


@pytest.mark.parametrize("verb", ["topk", "race"])
def test_sweep_across_the_band_has_no_jump(verb):
    """Adjacent points 0.02 apart in sharpness move by at most the rule
    difference (31bp) times the smoothstep's slope (1.5 per unit) times
    the step, plus the price's own slope: ~1bp, never a 31bp step."""
    sh = np.arange(_R1_BLEND_LO, _R1_BLEND_HI + 1e-9, 0.02)
    if verb == "topk":
        p = [top_k_probabilities(MU, 2, V=V, D=_D(10 / s), points=513)[0]
             for s in sh]
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = [race_probabilities(MU, V=V, D=_D(10 / s))[0] for s in sh]
    assert np.max(np.abs(np.diff(p))) < 2e-4


@pytest.mark.parametrize("sharp", [4.0, 6.0, 8.5, 8.99, 10.01, 10.5, 10.6,
                                   12.0, 40.0])
def test_rule_outside_the_band_is_unchanged(sharp):
    """Bitwise the old rule just outside the band's edges."""
    F, W = _rule(sharp)
    Fo, Wo = _old_rule(sharp)
    assert np.array_equal(F, Fo) and np.array_equal(W, Wo)


def test_band_meets_its_edges():
    """Each component's order is frozen at its own edge's value, so the
    rule tends to the edge rule from inside, and costs both together."""
    eps = 1e-6                       # 1 - w and w are then ~3e-12
    F, W = _rule(_R1_BLEND_LO + eps)
    Fg, Wg = _old_rule(_R1_BLEND_LO)            # GH 72, the rule at lo
    assert len(Wg) == 72 and len(W) == 72 + 81
    assert np.array_equal(F[:72], Fg) and np.allclose(W[:72], Wg, atol=1e-11)
    assert W[72:].sum() < 1e-11
    F, W = _rule(_R1_BLEND_HI - eps)
    Fm, Wm = _old_rule(_R1_BLEND_HI + eps)      # midpoint 81, just past hi
    assert len(Wm) == 81
    assert np.array_equal(F[72:], Fm) and np.allclose(W[72:], Wm, atol=1e-11)
    assert W[:72].sum() < 1e-11
    assert abs(W.sum() - 1.0) < 1e-14


def test_ports_carry_the_same_band():
    srcs = {
        "r/winning/R/races.R": (".R1_BLEND_LO <- 9.0", ".R1_BLEND_HI <- 10.0"),
        "julia/winning/src/winning.jl": ("const R1_BLEND_LO = 9.0",
                                         "const R1_BLEND_HI = 10.0"),
        "docs/js/winning/races.mjs": ("R1_BLEND_LO = 9.0",
                                      "R1_BLEND_HI = 10.0"),
    }
    assert (races._R1_BLEND_LO, races._R1_BLEND_HI) == (9.0, 10.0)
    for path, needles in srcs.items():
        text = open(os.path.join(ROOT, path)).read()
        for needle in needles:
            assert needle in text, (path, needle)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_band_matches_python():
    script = (
        "import { topKProbabilities } from './docs/js/winning/topk.mjs';\n"
        "const a = 10 / Math.SQRT2, V = [[-a], [a / 3], [a / 3], [a / 3]];\n"
        "const mu = [-0.7, -0.1, 0.2, 0.8];\n"
        "const out = [];\n"
        "for (const sd0 of [1 - 1e-7, 1, 1 + 1e-7, 10 / 8.9, 10 / 9.5])\n"
        "  out.push(topKProbabilities(mu, 2, { V, D: [sd0 * sd0, 1, 1, 1],"
        " points: 513 })[0]);\n"
        "console.log(JSON.stringify(out));\n")
    res = subprocess.run([NODE, "--input-type=module", "-e", script],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    import json
    js = np.array(json.loads(res.stdout))
    py = np.array([top_k_probabilities(MU, 2, V=V, D=_D(s), points=513)[0]
                   for s in (1 - 1e-7, 1, 1 + 1e-7, 10 / 8.9, 10 / 9.5)])
    assert abs(js[0] - js[1]) < 1e-7 and abs(js[2] - js[1]) < 1e-7
    np.testing.assert_allclose(js, py, atol=1e-9)
