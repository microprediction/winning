"""Round-10 race regressions: the guarded final Jacobi reprice (#823) and
the bounded open-bracket Newton step of the pair-mixture inverse (#830)."""
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from winning.factor.races import (_jacobi_sweeps, abilities_from_race,
                                  race_probabilities)
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

_MU823 = [-7.645826295483857, -4.355092679616064, 4.016859659459442,
          7.984059315640479]
_D823 = [0.028135701064085954, 0.02463857634420256, 2.010569296584627,
         0.028414577208752127]


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
    q = top_k_probabilities(_MU823, 3, D=_D823)
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
const mu = %s, D = %s;
const q = topKProbabilities(mu, 3, {D});
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
    got = _node(_JS823 % ("%s", "%s", json.dumps(_MU823), json.dumps(_D823)),
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
