"""Round-3 port regressions on the Python and browser side of the ports:
unit equivariance of the pair closed form, race inverse and block window
(#100), the classic normalizers (#339) and the compiled fastrace boundary
(#103, #339)."""
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from winning.factor import abilities_from_race, race_probabilities
from winning.factor.blocks import block_race_probabilities

from winning.classic.lattice import (as_classic_density, as_classic_prices,
                                     state_prices_from_offsets)
from winning.classic.lattice_calibration import solve_for_implied_offsets


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


def test_pair_closed_form_keeps_units():
    # the forward floored the contrast variance at 1e-300 (0.540 for an
    # exact 0.760 at c = 1e-151) and the inverse overflowed the variance
    # sum at D = 9e307 (#100)
    exact = float(ndtr(1 / np.sqrt(2)))
    for c in (1.0, 1e-150, 1e-151, 3e-154):
        p = race_probabilities(np.array([0.0, c]), V=np.zeros((2, 0)),
                               D=np.array([c * c, c * c]))
        assert abs(p[0] - exact) < 1e-15
    t = np.array([exact, 1 - exact])
    for c in (1e-151, 3e-154):
        m = abilities_from_race(t, D=np.array([c * c, c * c]))
        np.testing.assert_allclose(m / c, [-0.5, 0.5], atol=1e-12)
    ref = abilities_from_race(np.array([0.8, 0.2]), D=np.ones(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for d in (8e307, 9e307, 1e308):
            m = abilities_from_race(np.array([0.8, 0.2]), D=np.array([d, d]))
            assert np.all(np.isfinite(m))
            np.testing.assert_allclose(m / np.sqrt(d), ref, atol=1e-12)


def test_race_inverse_slope_floor_keeps_units():
    # an absolute -1e-6 slope floor stopped D = 1e20 0.062 short (#100)
    p = np.array([0.5, 0.3, 0.2])
    ref = abilities_from_race(p, D=np.ones(3), points=513)
    for d in (1e-20, 1e16, 1e20):
        m, info = abilities_from_race(p, D=np.full(3, d), points=513,
                                      return_info=True)
        assert info["converged"]
        np.testing.assert_allclose(m / np.sqrt(d), ref, atol=1e-7)


def test_block_window_has_no_absolute_floor():
    # 1e-14 units raised a 0.7% mass defect at 257 points (#100)
    mu = np.array([-1.0, -0.2, 0.4, 1.1])
    cl = np.array([0, 0, 1, 1])
    L = np.array([0.4, -0.2, 0.3, -0.1])
    D = np.array([0.7, 1.2, 0.8, 1.5])
    ref = block_race_probabilities(mu, cl, L, D, points=257, qa=9)
    for c in (1e-13, 1e-14, 1e-20):
        got = block_race_probabilities(c * mu, cl, c * L, c * c * D,
                                       points=257, qa=9)
        np.testing.assert_allclose(got, ref, atol=1e-12)


_JS = r"""
import { abilitiesFromRace } from %s;
import { blockRaceProbabilities } from %s;
console.warn = () => {};
const out = {};
const t = [0.7602499389065233, 0.2397500610934767];
out.tiny = [1e-151, 3e-154].map(c => abilitiesFromRace(t, {D: [c*c, c*c]}).map(v => v / c));
out.big = [9e307, 1e308].map(d => abilitiesFromRace([0.8, 0.2], {D: [d, d], points: 1001}).map(v => v / Math.sqrt(d)));
out.bigref = abilitiesFromRace([0.8, 0.2], {D: [1, 1], points: 1001});
out.ref = abilitiesFromRace([0.5, 0.3, 0.2], {D: [1, 1, 1], points: 513});
out.d20 = abilitiesFromRace([0.5, 0.3, 0.2], {D: [1e20, 1e20, 1e20], points: 513, returnInfo: true});
const mu = [-1, -0.2, 0.4, 1.1], cl = [0, 0, 1, 1], L = [0.4, -0.2, 0.3, -0.1], D = [0.7, 1.2, 0.8, 1.5];
out.blk = blockRaceProbabilities(mu, cl, L, D, {points: 257, qa: 9});
out.blk14 = blockRaceProbabilities(mu.map(v => 1e-14 * v), cl, L.map(v => 1e-14 * v), D.map(v => 1e-28 * v), {points: 257, qa: 9});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_pair_inverse_and_block_window_keep_units():
    mods = [Path(ROOT, "docs", "js", "winning", f).as_uri()
            for f in ("races.mjs", "blocks.mjs")]
    run = subprocess.run([NODE, "--input-type=module", "-e",
                          _JS % tuple(json.dumps(m) for m in mods)],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    got = json.loads(run.stdout)
    for row in got["tiny"]:                         # was +-3.5e-151 / c
        np.testing.assert_allclose(row, [-0.5, 0.5], atol=1e-8)
    for row in got["big"]:                          # was [-inf, inf]
        np.testing.assert_allclose(row, got["bigref"], atol=1e-12)
    assert got["d20"]["converged"]                  # was 0.062 short
    np.testing.assert_allclose(np.asarray(got["d20"]["mu"]) / 1e10,
                               got["ref"], atol=1e-7)
    np.testing.assert_allclose(got["blk14"], got["blk"], atol=1e-12)


def test_classic_normalizers_are_scale_safe():
    # finite atoms whose raw total overflows were normalised to zeros and
    # priced as a tie (#339)
    d = [0, 0, 1, 1, 0, 0, 0]
    huge = [0, 0, 1e308, 1e308, 0, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(as_classic_density(huge),
                                   as_classic_density(d), rtol=0, atol=0)
        np.testing.assert_allclose(state_prices_from_offsets(huge, [0, 1]),
                                   state_prices_from_offsets(d, [0, 1]),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(as_classic_prices([1e308, 9e307]),
                                   as_classic_prices([1.0, 0.9]),
                                   rtol=1e-15, atol=0)
    a = solve_for_implied_offsets([1e308, 9e307], d, nIter=10)
    b = solve_for_implied_offsets([1.0, 0.9], d, nIter=10)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert abs(a[0] - a[1]) > 1e-3          # not the tie it used to be
    # ordinary input is untouched, bit for bit
    raw = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]) * 3.0
    assert np.array_equal(as_classic_density(raw), raw / raw.sum())


try:
    import fastrace
except ImportError:                     # pure install: compiled tests skip
    fastrace = None
needs_fastrace = pytest.mark.skipif(fastrace is None,
                                    reason="fastrace not installed")


@needs_fastrace
@pytest.mark.parametrize("beta", [0.5, 0.7, 0.999])
def test_fastrace_refuses_exponential_power_below_one(beta):
    # the factory refuses beta < 1 (#103); the compiled export did not
    from scipy.special import gamma
    a = np.sqrt(gamma(1 / beta) / gamma(3 / beta))
    with pytest.raises(ValueError, match="domain"):
        fastrace.forward_and_slopes_base(
            np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
            np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 4, [beta, a])


@needs_fastrace
def test_fastrace_exponential_power_one_still_prices():
    from winning.factor.races import exponential_power_base
    b = exponential_power_base(1.0)
    beta, a = b.rust_base[1]
    p, _, _ = fastrace.forward_and_slopes_base(
        np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
        np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 4, [beta, a])
    assert np.all(np.isfinite(p))


@needs_fastrace
def test_fastrace_student_needs_unit_variance_nu():
    with pytest.raises(ValueError, match="domain"):
        fastrace.forward_and_slopes_base(
            np.array([-0.5, 1.1, -0.6]), np.zeros((3, 1)), np.ones(3),
            np.zeros((1, 1)), np.ones(1), 1001, -12.0, 12.0, 5, [1.5, 1.0])


@needs_fastrace
def test_fastrace_classic_density_needs_positive_mass():
    # seven zero atoms priced every runner at 0 (#339)
    with pytest.raises(ValueError, match="positive mass"):
        fastrace.classic_exact_state_prices([0.0] * 7, [0.0, 0.0])
    with pytest.raises(ValueError, match="positive mass"):
        fastrace.classic_exact_calibrate([0.0] * 7, [0.5, 0.5],
                                         [1.0, 0.0, -1.0], [0.0, 0.0], 3)
