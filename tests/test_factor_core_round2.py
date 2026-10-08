"""Regressions for the second factor-core issue batch. Each test names its
issue; each was run against the pre-fix tree and failed there."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.factor.topk import abilities_from_topk, top_k_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    return json.loads(out.stdout)


def _mod(name):
    # a file:// URI, JSON-quoted, so Windows paths are legal specifiers
    return json.dumps(Path(ROOT, "docs", "js", "winning", name).as_uri())


# ---------------------------------------------------------------- #583

MU_583 = np.array([-2.4085533796267504, -2.371749066692294,
                   0.34486967818371195, -1.761616714882222,
                   4.096640356222963, 2.100409126794591])
SD_583 = np.array([0.3622773278819501, 1.9392652513644468,
                   1.1350172886349525, 0.8731304646755176,
                   2.865406646646641, 0.6217100227308292])


def test_aitken_jump_is_a_checked_trial_583():
    """An exact top-1 target: the unchecked 539x extrapolation ended at
    logit residual 715 with abilities off by ~490."""
    D = SD_583 ** 2
    q = top_k_probabilities(MU_583, 1, D=D, points=257)
    fit, info = abilities_from_topk(q, 1, D=D, points=257, n_iter=120,
                                    tol=1e-8, return_info=True)
    err = fit - MU_583
    assert np.abs(err - err.mean()).max() < 1e-6
    assert info["max_logit_residual"] < 1e-5          # was 715
    rep = top_k_probabilities(fit, 1, D=D, points=257)
    assert np.abs(rep - q).max() < 1e-9


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_aitken_jump_is_a_checked_trial_583():
    got = _node(f"""
import {{ topKProbabilities, abilitiesFromTopk }} from {_mod("topk.mjs")};
console.warn = () => {{}};
const mu = {json.dumps(MU_583.tolist())};
const D = {json.dumps((SD_583 ** 2).tolist())};
const q = topKProbabilities(mu, 1, {{D, points: 257}});
const fit = abilitiesFromTopk(q, 1, {{D, points: 257, nIter: 120, tol: 1e-8,
                                     returnInfo: true}});
const rep = topKProbabilities(fit.mu, 1, {{D, points: 257}});
console.log(JSON.stringify({{mu: fit.mu, info: fit.info, rep, q}}));
""")
    err = np.array(got["mu"]) - MU_583
    assert np.abs(err - err.mean()).max() < 1e-6       # was ~900
    assert got["info"]["maxLogitResidual"] < 1e-5      # was 715
    assert np.abs(np.array(got["rep"]) - got["q"]).max() < 1e-9


# ---------------------------------------------------------------- #524

@pytest.mark.parametrize("d0", [0.003, 0.01, 0.03, 0.1])
def test_core_inverse_hazard_is_analytic_524(d0):
    """A narrow runner's density underflowed, was floored at 1e-300 while
    log S kept falling, and inf - inf made every ability NaN."""
    from winning.factor.core import (abilities_from_probabilities_factor,
                                     win_probabilities_factor)
    D = np.array([d0, 1.0, 1.0])
    V, F, W = np.zeros((3, 1)), np.zeros((1, 1)), np.ones(1)
    truth = np.array([0.5, 0.0, -0.5])
    target = win_probabilities_factor(truth, V, D, F, W)
    with np.errstate(over="raise", invalid="raise"):
        mu, info = abilities_from_probabilities_factor(
            target, V, D, F, W, return_info=True)
    assert np.isfinite(mu).all() and info["converged"]
    rep = win_probabilities_factor(mu, V, D, F, W)
    assert np.abs(rep - target).max() < 1e-6


# ---------------------------------------------------------------- #510

def _hier_510():
    from winning.factor.structures import Blocks, Nested, Tree
    c = np.array([0, 0, 1, 1])
    L = np.array([0.2, 0.3, -0.1, 0.4])
    g = np.array([0.1, -0.2, 0.3, -0.1])
    return [lambda D: Blocks(c, L, D),
            lambda D: Nested(c, L, D, g, 0.5),
            lambda D: Tree(c, L, D, [2, 2, -1], [0.3, 0.3, 0.2])]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_structured_inverse_reads_a_scalar_D_510(which):
    """structure_variances took len() of a 0-d D: the forward priced a
    scalar-D Blocks/Nested race and its own inverse raised TypeError."""
    from winning.factor.races import abilities_from_race, race_probabilities
    from winning.factor.structures import structure_variances
    make = _hier_510()[which]
    mu = np.array([-0.5, -0.1, 0.2, 0.4])
    mu -= mu.mean()
    s, v = make(1.0), make(np.ones(4))
    assert s.n == 4
    assert np.array_equal(structure_variances(s), structure_variances(v))
    p = race_probabilities(mu, structure=s, points=257)
    m, info = abilities_from_race(p, structure=s, points=257,
                                  return_info=True)
    mv = abilities_from_race(p, structure=v, points=257)
    assert info["converged"] and np.abs(m - mu).max() < 1e-7
    assert np.abs(m - mv).max() < 1e-12


# ---------------------------------------------------------------- #515

MU_515 = np.array([1.168799682867873, -0.8531654023507033,
                   -3.006600202361091, 1.7494547072828357,
                   0.9415112145610854])
SD_515 = np.array([1.1139780430874056, 0.9672196804840659,
                   0.30363853046872963, 3.389862172988461,
                   0.7137810123811725])


def test_adaptive_count_is_dyadic_515():
    from winning.factor.topk import _dyadic_points
    assert [_dyadic_points(n) for n in (2, 3, 513, 514, 610, 611, 1025,
                                        1026, 8192, 8193, 10 ** 6)] == \
        [2, 3, 513, 1025, 1025, 1025, 1025, 2049, 8193, 8193, 8193]


@pytest.mark.parametrize("points", [257, 513])
def test_topk_scale_jacobian_matches_the_public_map_515(points):
    """The auto count stepped 611 -> 610 under a 1e-5 change of one sd:
    the central difference read -0.307 against Jsigma -0.0029, and the
    loc/scale inverse stalled at 1.3e-5 on an exact target."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     top_k_jacobians)
    D, base = SD_515 ** 2, "gumbel"
    _, Js = top_k_jacobians(MU_515, 2, D=D, base=base, points=points)
    h = SD_515[2] * 1e-5
    sp, sm = SD_515.copy(), SD_515.copy()
    sp[2] += h
    sm[2] -= h
    fd = (top_k_probabilities(MU_515, 2, D=sp ** 2, base=base,
                              points=points)[2]
          - top_k_probabilities(MU_515, 2, D=sm ** 2, base=base,
                                points=points)[2]) / (2 * h)
    assert abs(fd - Js[2][2]) < 1e-6
    q1 = top_k_probabilities(MU_515, 1, D=D, base=base, points=points)
    q2 = top_k_probabilities(MU_515, 2, D=D, base=base, points=points)
    *_, info = loc_scale_from_topk_pair(q1, 1, q2, 2, base=base,
                                        points=points, n_iter=60, tol=1e-8,
                                        return_info=True)
    assert info["converged"], info


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_topk_scale_jacobian_matches_the_public_map_515():
    got = _node(f"""
import {{ topKProbabilities, topKJacobians, locScaleFromTopkPair,
         dyadicPoints }} from {_mod("topk.mjs")};
console.warn = () => {{}};
const mu = {json.dumps(MU_515.tolist())};
const sd = {json.dumps(SD_515.tolist())};
const D = sd.map(x => x * x), base = "gumbel", points = 513;
const {{ Jsigma }} = topKJacobians(mu, 2, {{ D, base, points }});
const h = sd[2] * 1e-5, sp = sd.slice(), sm = sd.slice();
sp[2] += h; sm[2] -= h;
const fd = (topKProbabilities(mu, 2, {{ D: sp.map(x => x * x), base, points }})[2]
  - topKProbabilities(mu, 2, {{ D: sm.map(x => x * x), base, points }})[2]) / (2 * h);
const q1 = topKProbabilities(mu, 1, {{ D, base, points }});
const q2 = topKProbabilities(mu, 2, {{ D, base, points }});
const fit = locScaleFromTopkPair(q1, 1, q2, 2, {{ base, points, nIter: 60,
                                                tol: 1e-8, returnInfo: true }});
console.log(JSON.stringify({{ an: Jsigma[2][2], fd, info: fit.info,
  dy: [514, 611, 1025, 1026, 9000].map(dyadicPoints) }}));
""")
    assert abs(got["fd"] - got["an"]) < 1e-6           # was 0.30 apart
    assert got["info"]["converged"], got["info"]
    assert got["dy"] == [1025, 1025, 1025, 2049, 8193]


# ---------------------------------------------------------------- #551

@pytest.mark.parametrize("bad", [np.inf, np.nan, 0.0, -1e-8, True, "1e-8",
                                 [1e-8]])
def test_inverses_refuse_a_tolerance_that_certifies_anything_551(bad):
    """tol=inf certified warm starts 15-24 points off as converged."""
    from winning.factor.races import abilities_from_race
    from winning.factor.topk import (abilities_from_rank_marginal,
                                     loc_scale_from_topk_pair)
    mu = np.array([-1.5, -0.4, 0.3, 1.6])
    D = np.array([0.25, 0.64, 1.44, 2.25])
    p1 = top_k_probabilities(mu, 1, D=D)
    p2 = top_k_probabilities(mu, 2, D=D)
    for call in (lambda: abilities_from_race(p1, D=D, tol=bad),
                 lambda: abilities_from_topk(p1, 1, D=D, tol=bad),
                 lambda: loc_scale_from_topk_pair(p1, 1, p2, 2, tol=bad),
                 lambda: abilities_from_rank_marginal(p1, 1, D=D, tol=bad)):
        with pytest.raises(ValueError, match="tol"):
            call()


# ---------------------------------------------------------------- #558

@pytest.mark.parametrize("bad", [-1.0, -np.inf, np.inf, np.nan, True,
                                 [0.0, 0.05]])
def test_loc_scale_refuses_an_invalid_ridge_558(bad):
    """A negative ridge was max(ridge, 0): the unregularized fit."""
    from winning.factor.topk import (loc_scale_from_topk_pair,
                                     loc_scale_from_win_and_second)
    mu = np.array([-0.7, -0.1, 0.2, 0.6])
    D = np.array([0.3, 0.8, 1.5, 2.0])
    q1 = top_k_probabilities(mu, 1, D=D)
    q2 = top_k_probabilities(mu, 2, D=D)
    with pytest.raises(ValueError, match="ridge"):
        loc_scale_from_topk_pair(q1, 1, q2, 2, ridge=bad)
    with pytest.raises(ValueError, match="ridge"):
        loc_scale_from_win_and_second(q1, q2 - q1, ridge=bad)


# ---------------------------------------------------------------- #590

def test_nan_mass_tol_is_refused_590():
    """NaN made `defect > mass_tol` false: an 18% defect renormalized."""
    from winning.factor.permutations import ordered_probabilities
    from winning.factor.races import removal_shares
    mu, D = [-5000.0, 0.0, 0.1, 0.2], [0.01] * 4
    for bad in (np.nan, -1.0, np.inf):
        with pytest.raises(ValueError, match="mass_tol"):
            removal_shares(mu, D=D, points=3, mass_tol=bad)
        with pytest.raises(ValueError, match="mass_tol"):
            ordered_probabilities(np.zeros(3), k=2, mass_tol=bad)


# ---------------------------------------------------------------- #523

@pytest.mark.parametrize("rust", [False, True])
def test_jvp_form_must_be_named_523(rust):
    """Anything but "grid" ran IBP -- 0.36 apart at L = 11."""
    import winning
    from winning.factor.core import jacobian_vector_product
    mu = np.array([-1.0, 0.2, 1.8])
    V = np.array([[1.2], [-0.6], [0.3]])
    D = np.array([0.15, 1.2, 2.5])
    F = np.array([[-1.2], [0.4], [2.1]])
    W = np.array([0.2, 0.5, 0.3])
    h = np.array([0.4, -0.3, 0.1])
    prev = winning.use_rust(rust)
    try:
        for bad in ("typo", "", "GRID", None):
            with pytest.raises(ValueError, match="form"):
                jacobian_vector_product(mu, V, D, F, W, h, points=11,
                                        form=bad)
        a = jacobian_vector_product(mu, V, D, F, W, h, points=11, form="ibp")
        b = jacobian_vector_product(mu, V, D, F, W, h, points=11, form="grid")
        assert np.abs(a - b).max() > 0.1
    finally:
        winning.use_rust(prev)


# ---------------------------------------------------------------- #582

def test_race_window_must_be_named_582():
    """window="bulkk" silently meant span: 39 points off the bulk price."""
    from winning.factor.races import race_probabilities
    mu, D = np.array([0.0, 1.0, 2.0]), np.array([0.01, 4.0, 100.0])
    for bad in ("bulkk", "", "Bulk", None, 3):
        with pytest.raises(ValueError, match="window"):
            race_probabilities(mu, D=D, points=257, window=bad)
    a = race_probabilities(mu, D=D, points=257, window="span")
    b = race_probabilities(mu, D=D, points=257)
    assert np.abs(a - b).max() > 0.3


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_race_window_must_be_named_582():
    got = _node(f"""
import {{ raceProbabilities }} from {_mod("races.mjs")};
console.warn = () => {{}};
const out = [];
for (const w of ["bulkk", "", null, 3]) {{
  try {{ raceProbabilities([0, 1, 2], {{ D: [0.01, 4, 100], window: w }});
        out.push("accepted"); }}
  catch (e) {{ out.push(e.message.includes("window") ? "refused" : e.message); }}
}}
console.log(JSON.stringify(out));
""")
    assert got == ["refused"] * 4


# ---------------------------------------------------------------- #538

def test_core_inverse_budget_exhaustion_is_reported_538():
    """Exhaustion returned silently; n_iter=0 with return_info raised
    UnboundLocalError."""
    import warnings
    from winning.factor.core import (abilities_from_probabilities_factor,
                                     hermite_nodes, win_probabilities_factor)
    mu = np.array([-2.0, -0.2, 1.1, 2.4])
    V = np.array([[-0.8], [-0.1], [0.3], [0.9]])
    D = np.array([0.35, 0.8, 0.5, 1.2])
    F, W = hermite_nodes(1, Q=11)
    t = win_probabilities_factor(mu, V, D, F, W, points=1001)
    for budget in (0, 1, 2):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            m = abilities_from_probabilities_factor(t, V, D, F, W,
                                                    n_iter=budget,
                                                    points=1001)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m2, info = abilities_from_probabilities_factor(
                t, V, D, F, W, n_iter=budget, points=1001, return_info=True)
        assert np.array_equal(m, m2)
        assert info["iterations"] == budget and not info["converged"]
        rep = win_probabilities_factor(m, V, D, F, W, points=1001)
        assert abs(np.abs(np.log(rep) - np.log(t)).max()
                   - info["residual"]) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m, info = abilities_from_probabilities_factor(t, V, D, F, W,
                                                      points=1001,
                                                      return_info=True)
    assert info["converged"]
    for bad in (-1, 1.5, np.inf, True):
        with pytest.raises(ValueError, match="n_iter"):
            abilities_from_probabilities_factor(t, V, D, F, W, n_iter=bad)


# ---------------------------------------------------------------- #592

def test_target_floor_is_a_probability_not_a_unit_592():
    """Floored before normalizing, [c, 0] gave gaps 0 / 4.75 / 7.03 at
    c = 1e-6 / 1 / 1e6."""
    from winning.factor.races import abilities_from_race
    race = [abilities_from_race(np.array([c, 0.0]), D=np.array([0.5, 0.5]),
                                target_floor=1e-6, return_info=True)
            for c in (1e-6, 1.0, 1e6, 1e308)]
    for m, info in race[1:]:
        assert np.abs(m - race[0][0]).max() < 1e-12
        assert np.array_equal(info["floored"], race[0][1]["floored"])
    tk = [abilities_from_topk(np.array([0.7, 0.3, 0.0]) * c, 1,
                              D=np.ones(3), target_floor=1e-4, n_iter=100,
                              return_info=True)
          for c in (1e-3, 1.0, 1e3)]
    for m, info in tk:
        assert info["converged"]
        assert np.abs(m - tk[0][0]).max() < 1e-9
    with pytest.raises(ValueError, match="non-negative"):
        abilities_from_race(np.array([0.5, -0.1, 0.6]), D=np.ones(3),
                            target_floor=1e-6)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_target_floor_is_a_probability_592():
    got = _node(f"""
import {{ abilitiesFromRace }} from {_mod("races.mjs")};
import {{ abilitiesFromTopk }} from {_mod("topk.mjs")};
console.warn = () => {{}};
const race = [1e-6, 1, 1e6].map(c => abilitiesFromRace([c, 0],
  {{ D: [0.5, 0.5], targetFloor: 1e-6 }}));
const topk = [1e-3, 1, 1e3].map(c => abilitiesFromTopk([0.7 * c, 0.3 * c, 0],
  1, {{ D: [1, 1, 1], targetFloor: 1e-4, nIter: 100 }}));
console.log(JSON.stringify({{ race, topk }}));
""")
    for key, tol in (("race", 1e-12), ("topk", 1e-9)):
        ref = np.array(got[key][0])
        for m in got[key][1:]:
            assert np.abs(np.array(m) - ref).max() < tol


# ---------------------------------------------------------------- #497

@pytest.mark.parametrize("n_iter", [0, 1, 2])
def test_inverse_diagnostics_describe_the_returned_iterate_497(n_iter):
    """The residual came from the iterate before the last step (0.497
    reported, 0.071 actual at n_iter=1), and inf at n_iter=0."""
    from winning.factor.races import abilities_from_race, race_probabilities
    t = np.array([0.7, 0.2, 0.1])
    mu, info = abilities_from_race(t, D=np.ones(3), points=257,
                                   n_iter=n_iter, return_info=True)
    actual = np.abs(np.log(race_probabilities(mu, D=np.ones(3), points=257))
                    - np.log(t)).max()
    assert abs(info["max_log_residual"] - actual) < 1e-12
    mk, ik = abilities_from_topk(t, 1, D=np.ones(3), points=257,
                                 n_iter=n_iter, return_info=True)
    q = top_k_probabilities(mk, 1, D=np.ones(3), points=257)
    actual = np.abs((np.log(q) - np.log1p(-q))
                    - (np.log(t) - np.log1p(-t))).max()
    assert abs(ik["max_logit_residual"] - actual) < 1e-12


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_topk_diagnostics_describe_the_returned_iterate_497():
    got = _node(f"""
import {{ abilitiesFromTopk, topKProbabilities }} from {_mod("topk.mjs")};
const t = [0.7, 0.2, 0.1], out = [];
for (const n of [1, 2]) {{
  const r = abilitiesFromTopk(t, 1, {{ D: [1, 1, 1], points: 257, nIter: n,
                                      returnInfo: true }});
  const q = topKProbabilities(r.mu, 1, {{ D: [1, 1, 1], points: 257 }});
  out.push([r.info.maxLogitResidual, Math.max(...q.map((v, i) =>
    Math.abs(Math.log(v) - Math.log1p(-v) - Math.log(t[i]) + Math.log1p(-t[i]))))]);
}}
console.log(JSON.stringify(out));
""")
    for reported, actual in got:
        assert abs(reported - actual) < 1e-12


# ---------------------------------------------------------------- #508

def test_rank_zero_block_loadings_are_the_independent_race_508():
    """An (n, 0) loading fell into the rank-one reshape and raised."""
    from winning.factor import race_probabilities
    from winning.factor.blocks import (block_race_jacobian,
                                       block_race_probabilities,
                                       nested_race_probabilities)
    from winning.factor.structures import Blocks
    mu = np.array([-0.5, -0.1, 0.2, 0.4])
    D = np.array([0.8, 0.9, 1.0, 1.1])
    c = np.array([0, 0, 1, 1])
    V0, Z = np.empty((4, 0)), np.zeros(4)
    ref = block_race_probabilities(mu, c, Z, D, points=1025)
    assert np.abs(block_race_probabilities(mu, c, V0, D, points=1025)
                  - ref).max() < 1e-15
    assert np.abs(race_probabilities(mu, structure=Blocks(c, V0, D),
                                     points=1025) - ref).max() < 1e-15
    assert np.abs(block_race_jacobian(mu, c, V0, D, points=1025)
                  - block_race_jacobian(mu, c, Z, D, points=1025)).max() == 0
    g = np.array([0.2, -0.1, 0.3, 0.0])
    assert np.abs(nested_race_probabilities(mu, c, V0, D, coupling=g)
                  - nested_race_probabilities(mu, c, Z, D,
                                              coupling=g)).max() < 1e-15


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_rank_zero_block_loadings_take_the_rank_one_path_508():
    got = _node(f"""
import {{ blockRaceProbabilities }} from {_mod("blocks.mjs")};
const mu = [-0.5, -0.1, 0.2, 0.4], c = [0, 0, 1, 1], D = [0.8, 0.9, 1.0, 1.1];
const t0 = Date.now();
const a = blockRaceProbabilities(mu, c, [[], [], [], []], D);
const ms = Date.now() - t0;
const b = blockRaceProbabilities(mu, c, [0, 0, 0, 0], D);
console.log(JSON.stringify({{ d: Math.max(...a.map((v, i) => Math.abs(v - b[i]))), ms }}));
""")
    assert got["d"] < 1e-15
    assert got["ms"] < 500                       # was ~2200 ms


# ------------------------------------------------- #461 #463 #483

def test_inverse_normalizers_survive_a_sum_overflow_461_463_483():
    """[0.8, 0.6, 0.4, 0.2] * 1e308: every entry finite, the sum not; the
    target collapsed to NaN before the solver ran."""
    import warnings
    from winning.factor.blocks import abilities_from_block_race
    from winning.factor.topk import (abilities_from_rank_marginal,
                                     loc_scale_from_win_and_second,
                                     rank_probabilities)
    from winning.shapes import rescaled_target
    q = np.array([0.8, 0.6, 0.4, 0.2])
    t = q / q.sum()
    assert np.array_equal(rescaled_target(q), t)       # bit-identical path
    big = q * 1e308
    assert np.abs(rescaled_target(big) - t).max() < 1e-15
    mu = np.array([-0.5, -0.1, 0.2, 0.4])
    R = rank_probabilities(mu, D=np.ones(4))
    w, s = R[:, 0], R[:, 1]
    calls = [
        lambda c: abilities_from_topk(q * c, 2, points=257),
        lambda c: abilities_from_rank_marginal(q * c, 1, points=257,
                                               n_iter=40),
        lambda c: abilities_from_block_race(
            q * c, [0, 0, 1, 1], [0.25, -0.15, 0.35, -0.05],
            [0.8, 0.9, 1.0, 1.1], points=129)[0],
        lambda c: loc_scale_from_win_and_second(w / w.max() * c,
                                                s / s.max() * c)[0],
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for call in calls:
            assert np.abs(call(1e308) - call(1.0)).max() < 1e-9
