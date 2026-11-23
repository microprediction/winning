"""Round-10 race regressions: the guarded final Jacobi reprice (#823) and
the bounded open-bracket Newton step of the pair-mixture inverse (#830),
the exported normal callable taking the named dispatch (#858), and the
finite nonempty door of the closed-form Luce helpers (#799)."""
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from winning.factor.permutations import plackett_luce_prefix_logprob
from winning.factor.races import (BASES, _jacobi_sweeps, abilities_from_race,
                                  abilities_from_softmax,
                                  plackett_luce_order_logprob,
                                  race_probabilities, softmax_probabilities)
from winning.factor.topk import abilities_from_topk, top_k_probabilities

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


def _node(js, *mods):
    uris = tuple(json.dumps(Path(ROOT, "docs", "js", "winning", m).as_uri())
                 for m in mods)
    run = subprocess.run([NODE, "--input-type=module", "-e", js % uris],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    return json.loads(run.stdout)


# -- #823: the exhaustion reprice is guarded like the in-loop forward -------

_D823 = [0.028135701064085954, 0.02463857634420256, 2.010569296584627,
         0.028414577208752127]
# the issue's exact target, as the forward prints it on the reporting
# platform; regenerating it elsewhere can round a membership to 1
_Q823 = [0.9999999999999999, 0.9999999999999999, 0.9972676865924252,
         0.0027323134075749387]


def _refusing(m):
    if np.abs(m).max() > 0.5:
        raise RuntimeError("unpriceable iterate")
    return np.array([1.0, -1.0]), np.array([-1.0, -1.0])


def test_final_reprice_refusal_returns_last_priced_iterate():
    # one permitted step lands on an unpriceable iterate; the exhaustion
    # reprice raised instead of returning the priced start (#823)
    mu, conv, rmax, iters = _jacobi_sweeps(np.zeros(2), _refusing, 1.0, 1.0,
                                           1, 1e-8)
    np.testing.assert_array_equal(mu, [0.0, 0.0])
    assert conv is False and rmax == 1.0 and iters == 1
    # with nothing ever priced the refusal still propagates
    with pytest.raises(RuntimeError):
        _jacobi_sweeps(np.ones(2), _refusing, 1.0, 1.0, 1, 1e-8)


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def test_topk_inverse_does_not_throw_on_its_budget():
    # n_iter = 3 and 4 raised "membership above one" (#823)
    q = np.array(_Q823)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 4, 5):
            mu, info = abilities_from_topk(q, 3, D=_D823, n_iter=n,
                                           return_info=True)
            assert not info["converged"]
            back = top_k_probabilities(mu, 3, D=_D823)
            got = np.abs(_logit(back) - _logit(q)).max()
            assert abs(got - info["max_logit_residual"]) < 1e-9


def test_ordinary_topk_inverse_unaffected_by_823_guard():
    # an ordinary target converges inside the loop: the guard is never hit
    mu0 = np.array([-0.6, -0.1, 0.2, 0.5])
    q = top_k_probabilities(mu0, 2)
    mu, info = abilities_from_topk(q, 2, return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(mu, mu0 - mu0.mean(), atol=1e-6)


_JS823 = r"""
import { topKProbabilities, abilitiesFromTopk } from %s;
import { jacobiSweeps } from %s;
console.warn = () => {};
const q = %s, D = %s;
const out = {q, fits: []};
for (const nIter of [2, 3, 4, 5]) {
  try {
    const f = abilitiesFromTopk(q, 3, {D, nIter, returnInfo: true});
    out.fits.push({...f, back: topKProbabilities(f.mu, 3, {D})});
  } catch (e) { out.fits.push({threw: e.message}); }
}
const fwd = m => {
  if (Math.max(...m.map(Math.abs)) > 0.5) throw new Error("unpriceable");
  return {resid: [1, -1], dres: [-1, -1]};
};
out.helper = jacobiSweeps([0, 0], fwd, 1, 1, 1, 1e-8);
try { jacobiSweeps([1, 1], fwd, 1, 1, 1, 1e-8); out.cold = "returned"; }
catch (e) { out.cold = "threw"; }
const q2 = topKProbabilities([-0.6, -0.1, 0.2, 0.5], 2);
out.ordinary = abilitiesFromTopk(q2, 2, {returnInfo: true});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_final_reprice_is_guarded():
    got = _node(_JS823 % ("%s", "%s", json.dumps(_Q823), json.dumps(_D823)),
                "topk.mjs", "core.mjs")
    q = np.asarray(got["q"])
    for f in got["fits"]:
        assert "threw" not in f, f
        assert f["info"]["converged"] is False
        resid = np.abs(_logit(f["back"]) - _logit(q)).max()
        assert abs(resid - f["info"]["maxLogitResidual"]) < 1e-9
    h = got["helper"]
    assert h["mu"] == [0, 0] and h["converged"] is False
    assert h["residMax"] == 1 and h["iterations"] == 1
    assert got["cold"] == "threw"
    assert got["ordinary"]["info"]["converged"]


# -- #830: the pair-mixture Newton step is capped while a side is open ------

_V830 = [[1.0], [0.0]]
_F830 = [[10.0], [0.0], [-10.0]]
_W830 = [0.12, 0.72, 0.16]


def _pair_fit(D):
    kw = dict(V=_V830, D=[D, D], F=_F830, W=_W830)
    t = race_probabilities([-0.025, 0.025], **kw)
    mu, info = abilities_from_race(t, return_info=True, **kw)
    back = race_probabilities(mu, **kw)
    return mu, info, float(np.abs(np.log(back) - np.log(t)).max())


def test_pair_mixture_inverse_recovers_plateau_target():
    # a ~1e-66 derivative ratio on the 0.12 plateau proposed a -6.7e64
    # step that the open bracket accepted; the scalar solve then failed
    # and the generic fallback returned +-0.873, residual 1.05 (#830)
    mu, info, resid = _pair_fit(0.005)
    assert info["converged"] and info["iterations"] == 0
    np.testing.assert_allclose(mu, [-0.025, 0.025], atol=1e-12)
    assert resid < 1e-8


def test_pair_mixture_inverse_ordinary_neighbours():
    # the D the scalar solve already handled: still exact, still direct
    for D in (0.05, 0.5, 2.0):
        mu, info, resid = _pair_fit(D)
        assert info["converged"] and info["iterations"] == 0
        np.testing.assert_allclose(mu, [-0.025, 0.025], atol=1e-12)
        assert resid < 1e-12


_JS830 = r"""
import { raceProbabilities, abilitiesFromRace } from %s;
console.warn = () => {};
const V = [[1], [0]], F = [[10], [0], [-10]], W = [0.12, 0.72, 0.16];
const out = [];
for (const d of [0.005, 0.05, 0.5, 2]) {
  const D = [d, d];
  const t = raceProbabilities([-0.025, 0.025], {V, D, F, W});
  const f = abilitiesFromRace(t, {V, D, F, W, returnInfo: true});
  const b = raceProbabilities(f.mu, {V, D, F, W});
  out.push({...f, resid: Math.max(...b.map((x, i) => Math.abs(Math.log(x / t[i]))))});
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_pair_mixture_inverse_recovers_plateau_target():
    for f in _node(_JS830, "races.mjs"):     # D = 0.005 gave +-0.873
        assert f["converged"] and f["iterations"] == 0
        np.testing.assert_allclose(f["mu"], [-0.025, 0.025], atol=1e-12)
        assert f["resid"] < 1e-8


# -- #858: BASES["normal"] is the named law, so it takes the exact pair path -

@pytest.mark.parametrize("gap", [20.0, 33.0, 40.0])
@pytest.mark.parametrize("flip", [False, True])
def test_normal_callable_takes_the_pair_closed_form(gap, flip):
    # the callable went to the generic lattice: gap 40 came back +-46.07,
    # residual 286.5, not converged, against an exact +-20 (#858)
    truth = np.array([gap / 2, -gap / 2] if flip else [-gap / 2, gap / 2])
    t = race_probabilities(truth, D=[1, 1], base=BASES["normal"])
    np.testing.assert_array_equal(t, race_probabilities(truth, D=[1, 1]))
    named = abilities_from_race(t, D=[1, 1], return_info=True)
    called = abilities_from_race(t, D=[1, 1], base=BASES["normal"],
                                 return_info=True)
    np.testing.assert_array_equal(called[0], named[0])
    assert called[1]["converged"] and called[1]["iterations"] == 0
    np.testing.assert_allclose(called[0], truth, atol=1e-6)


def test_normal_callable_matches_named_on_ordinary_fields():
    # a three-runner field and a factor pair: same answer either way
    mu = np.array([-0.4, 0.1, 0.3])
    np.testing.assert_array_equal(
        race_probabilities(mu, D=[1, 0.5, 2], base=BASES["normal"]),
        race_probabilities(mu, D=[1, 0.5, 2]))
    p = np.array([0.5, 0.3, 0.2])
    np.testing.assert_array_equal(
        abilities_from_race(p, D=[1, 0.5, 2], base=BASES["normal"]),
        abilities_from_race(p, D=[1, 0.5, 2]))
    kw = dict(V=[[0.5], [-0.3]], D=[0.6, 0.9])
    np.testing.assert_array_equal(
        abilities_from_race([0.7, 0.3], base=BASES["normal"], **kw),
        abilities_from_race([0.7, 0.3], **kw))


_JS858 = r"""
import { BASES, raceProbabilities, abilitiesFromRace } from %s;
console.warn = () => {};
const out = [];
for (const gap of [20, 33, 40]) for (const flip of [false, true]) {
  const truth = flip ? [gap / 2, -gap / 2] : [-gap / 2, gap / 2];
  const t = raceProbabilities(truth, {D: [1, 1], base: BASES.normal});
  const tn = raceProbabilities(truth, {D: [1, 1]});
  const named = abilitiesFromRace(t, {D: [1, 1], returnInfo: true});
  const called = abilitiesFromRace(t, {D: [1, 1], base: BASES.normal, returnInfo: true});
  out.push({truth, t, tn, named, called});
}
const p3 = [0.5, 0.3, 0.2], D3 = [1, 0.5, 2];
const ord = {
  f: [raceProbabilities([-0.4, 0.1, 0.3], {D: D3, base: BASES.normal}),
      raceProbabilities([-0.4, 0.1, 0.3], {D: D3})],
  i: [abilitiesFromRace(p3, {D: D3, base: BASES.normal}), abilitiesFromRace(p3, {D: D3})],
};
console.log(JSON.stringify({out, ord}));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_normal_callable_takes_the_pair_closed_form():
    got = _node(_JS858, "races.mjs")
    for r in got["out"]:
        assert r["t"] == r["tn"]
        assert r["called"]["mu"] == r["named"]["mu"]
        assert r["called"]["converged"] and r["called"]["iterations"] == 0
        np.testing.assert_allclose(r["called"]["mu"], r["truth"], atol=1e-6)
    assert got["ord"]["f"][0] == got["ord"]["f"][1]
    assert got["ord"]["i"][0] == got["ord"]["i"][1]


# -- #799: the closed-form Luce helpers refuse empty and non-finite input ---

_BAD = [np.inf, -np.inf, np.nan]
_VF = dict(V=[[0.3], [-0.1], [0.2]], F=[[-1.0], [1.0]], W=[0.5, 0.5])


@pytest.mark.parametrize("bad", _BAD)
def test_luce_helpers_refuse_non_finite_abilities(bad):
    # [0, inf, 1] priced [0.73, 0, 0.27]; the order logprob of [0, inf]
    # was 0.0 (certainty); NaN came back as NaN (#799)
    mu = [0.0, bad, 1.0]
    for kw in ({}, _VF):
        with pytest.raises(ValueError, match="not finite"):
            softmax_probabilities(mu, **kw)
        with pytest.raises(ValueError, match="not finite"):
            plackett_luce_order_logprob(mu, [0, 1, 2], **kw)
        with pytest.raises(ValueError, match="not finite"):
            plackett_luce_prefix_logprob(mu, [0, 1], **kw)
    with pytest.raises(ValueError, match="non-finite"):
        abilities_from_softmax([0.2, bad, 0.8])


def test_luce_helpers_refuse_empty_input():
    # abilities_from_softmax([]) returned [] (#799)
    with pytest.raises(ValueError, match="nonempty"):
        abilities_from_softmax([])
    with pytest.raises(ValueError, match="empty"):
        softmax_probabilities([])
    with pytest.raises(ValueError, match="empty"):
        plackett_luce_order_logprob([], [])


def test_luce_helpers_ordinary_inputs_unchanged():
    mu = np.array([0.0, 0.5, 1.0])
    z = np.exp(-mu)
    np.testing.assert_allclose(softmax_probabilities(mu), z / z.sum(),
                               rtol=1e-15)
    p = softmax_probabilities(mu)
    np.testing.assert_allclose(abilities_from_softmax(p), mu - mu.mean(),
                               atol=1e-15)
    lp = np.log(z[0] / z.sum()) + np.log(z[1] / (z[1] + z[2]))
    assert abs(plackett_luce_order_logprob(mu, [0, 1, 2]) - lp) < 1e-14
    assert abs(plackett_luce_prefix_logprob(mu, [0, 1]) - lp) < 1e-14
    assert np.isfinite(softmax_probabilities(mu, **_VF)).all()
