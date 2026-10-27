"""The rank-one Gauss-Hermite rule is continuous in sharpness (#850).

Below the Hermite/midpoint band (#783) the rank-one default rule was
Gauss-Hermite of order ceil(8 sharp). GH prices alternate with the parity
of the order, so the price jumped at every 1/8 of sharpness: by 12bp at
sharpness 4, 36bp at 6 and 59bp at 9 on #783's field. The rule now
interpolates adjacent orders up to order 24 and then uses the
parity-balanced pair GH(m-1)/2 + GH(m)/2, which also removes most of the
error the alternation carried. Outside (15/8, 9) the rule is unchanged.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
from scipy.special import ndtri

from winning.factor import races
from winning.factor.core import hermite_nodes
from winning.factor.races import _setup, race_probabilities
from winning.factor.topk import top_k_probabilities
from winning.shapes import as_factor_law

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")

A = 10 / np.sqrt(2)
V783 = np.array([[-A], [A / 3], [A / 3], [A / 3]])   # sharpness 10 / sd0
MU = np.array([-0.7, -0.1, 0.2, 0.8])
# a field that reaches every sharpness: loadings scaled, D = 1
_v = np.array([1.0, -0.4, 0.3, -0.9])
_v -= _v.mean()


def _D(sd0):
    return np.array([sd0 * sd0, 1.0, 1.0, 1.0])


def _scaled(sharp):
    return (_v * sharp / (np.sqrt(2) * np.abs(_v).max()))[:, None]


def _rule(sharp):
    _, _, _, F, W, _, _, _ = _setup(MU, _scaled(sharp), np.ones(4), None,
                                    None, "normal")
    return F, W


def _old_rule(sharp):
    """The pre-#850 rank-one Gauss-Hermite dispatch."""
    F, W = hermite_nodes(1, Q=int(np.clip(np.ceil(8.0 * sharp), 15, 201)))
    return as_factor_law(F, W, negligible=False)


@pytest.mark.parametrize("sharp", [4.0, 6.0, 71 / 8])
def test_issue_fixture_has_no_jump(sharp):
    """#783's field across an order change: 12, 36 and 59bp on main."""
    sd0 = 10.0 / sharp
    lo = top_k_probabilities(MU, 2, V=V783, D=_D(sd0 * (1 + 1e-9)),
                             points=513)
    hi = top_k_probabilities(MU, 2, V=V783, D=_D(sd0 * (1 - 1e-9)),
                             points=513)
    assert np.max(np.abs(hi - lo)) < 1e-7


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # GH(72)'s wide nodes
@pytest.mark.parametrize("verb", ["topk", "race"])
def test_no_jump_at_any_order_change(verb):
    """Every place the old order changed, k/8 for k = 16..72, straddled
    by a relative 1e-10 in sharpness. Main jumps by up to 1e-2 here."""
    def price(s):
        if verb == "topk":
            return top_k_probabilities(MU, 2, V=_scaled(s), D=np.ones(4))
        return race_probabilities(MU, V=_scaled(s), D=np.ones(4))
    worst = 0.0
    for k in range(16, 73, 2):
        s = k / 8
        worst = max(worst, np.max(np.abs(price(s * (1 + 1e-10))
                                         - price(s * (1 - 1e-10)))))
    assert worst < 1e-8


@pytest.mark.parametrize("sharp", [1.0, 1.5, 15 / 8, 2.0, 2.5, 3.0, 9.0])
def test_rule_is_unchanged_where_it_was_continuous(sharp):
    """Bitwise the old rule at and below 15/8, at every integer 8 sharp up
    to 24 (where the interpolation sits on an old order), and at 9, the
    band's lower edge."""
    F, W = _rule(sharp)
    Fo, Wo = _old_rule(sharp)
    assert np.array_equal(F, Fo) and np.array_equal(W, Wo)


def test_rule_shape_and_cost():
    """Weights sum to one; nodes never exceed twice the old rule's."""
    for s in np.arange(1.9, 9.0, 1 / 64):
        F, W = _rule(s)
        assert abs(W.sum() - 1.0) < 1e-13
        assert len(W) <= 2 * len(_old_rule(s)[1]), s
    F, W = _rule(5.0)                    # the parity pair, y = 25
    assert len(W) == 24 + 25
    F, W = _rule(8.0)                    # held at the cap
    assert len(W) == 33 + 34


def _ref(mu, V, D):
    Q = 4001
    F = ndtri((np.arange(Q) + 0.5) / Q)[:, None]
    return top_k_probabilities(mu, 2, V=V, D=D, F=F, W=np.full(Q, 1.0 / Q))


@pytest.mark.parametrize("sharp", [5.0, 7.0, 8.5])
def test_accuracy_against_a_fine_reference(sharp):
    """The parity-balanced pair against a 4001-node midpoint rule: 3.2e-4,
    3.3e-4 and 4.8e-4 off. The old GH(ceil 8 sharp) was 1.1-1.3e-3,
    2.3-2.4e-3 and 2.8-2.9e-3 off, by the parity of the order."""
    tol = 6e-4
    D = _D(10.0 / sharp)
    got = top_k_probabilities(MU, 2, V=V783, D=D)
    assert np.max(np.abs(got - _ref(MU, V783, D))) < tol


def test_ports_carry_the_same_rule():
    srcs = {
        "r/winning/R/races.R": (".R1_PAIR_FROM <- 24",
                                ".R1_PAIR_PER_SHARP <- 5.0",
                                ".R1_PAIR_CAP <- 34",
                                "r == 1 && 8 * sharp > 15"),
        "julia/winning/src/winning.jl": ("const R1_PAIR_FROM = 24",
                                         "const R1_PAIR_PER_SHARP = 5.0",
                                         "const R1_PAIR_CAP = 34",
                                         "r == 1 && 8 * sharp > 15"),
        "docs/js/winning/races.mjs": ("R1_PAIR_FROM = 24",
                                      "R1_PAIR_PER_SHARP = 5.0",
                                      "R1_PAIR_CAP = 34",
                                      "r === 1 && 8 * sharp > 15"),
    }
    assert (races._R1_PAIR_FROM, races._R1_PAIR_PER_SHARP,
            races._R1_PAIR_CAP) == (24, 5.0, 34)
    for path, needles in srcs.items():
        text = open(os.path.join(ROOT, path)).read()
        for needle in needles:
            assert needle in text, (path, needle)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_rule_matches_python():
    sharps = [2.06, 3.1, 4.86, 6.9, 8.9, 71 / 8 * (1 - 1e-10),
              71 / 8 * (1 + 1e-10)]
    script = (
        "import { topKProbabilities } from './docs/js/winning/topk.mjs';\n"
        f"const v = {_v.tolist()}, mu = {MU.tolist()}, "
        f"nv = {float(np.abs(_v).max())};\n"
        "const out = [];\n"
        f"for (const s of {sharps})\n"
        "  out.push(topKProbabilities(mu, 2, { V: v.map(x => [x * s /"
        " (Math.SQRT2 * nv)]), D: [1, 1, 1, 1] })[0]);\n"
        "console.log(JSON.stringify(out));\n")
    res = subprocess.run([NODE, "--input-type=module", "-e", script],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    import json
    js = np.array(json.loads(res.stdout))
    py = np.array([top_k_probabilities(MU, 2, V=_scaled(s),
                                       D=np.ones(4))[0] for s in sharps])
    np.testing.assert_allclose(js, py, atol=1e-9)
