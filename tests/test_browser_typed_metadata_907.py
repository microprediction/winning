"""Callable-base span and kink metadata may be plain or typed arrays (#907).

The browser accepted base.span only through Array.isArray, so a
Float64Array([25, 8]) was read as "no span" and the lattice fell back to
[12, 12], truncating a rare mode: a 0.005 win price came back 5e-324. The
ordered-market inverse read base.kinks the same way, so a typed kink list
made a Laplace base smooth and the #267 split-lattice repair never ran.
Python's lookup tested bool() of the kinks, which a one-element numpy
array holding 0.0 fails.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from winning.factor.races import BASES, _base_kinks, race_probabilities

NODE = shutil.which("node")
pytestmark_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                         check=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


MIXTURE = """
import { ndtr, logndtr, npdf } from "./docs/js/winning/core.mjs";
import { raceProbabilities } from "./docs/js/winning/races.mjs";
const w = [0.005, 0.995];
const m = [-14.1053184309531, 0.07088099714046785];
const s = [0.01417619942809357, 0.01417619942809357];
const lse = xs => { const a = Math.max(...xs); return a + Math.log(xs.reduce((v, x) => v + Math.exp(x - a), 0)); };
export function makeBase(span) {
  const base = z => {
    let S = 0, f = 0, fp = 0;
    for (let j = 0; j < 2; ++j) {
      const u = (z - m[j]) / s[j], p = npdf(u);
      S += w[j] * ndtr(-u); f += w[j] * p / s[j]; fp -= w[j] * u * p / (s[j] * s[j]);
    }
    return [Math.max(S, 1e-300), f, fp];
  };
  base.cdf = z => w.reduce((a, q, j) => a + q * ndtr((z - m[j]) / s[j]), 0);
  base.logCdf = z => lse(w.map((q, j) => Math.log(q) + logndtr((z - m[j]) / s[j])));
  base.logSf = z => lse(w.map((q, j) => Math.log(q) + logndtr((m[j] - z) / s[j])));
  base.resolution = s[0];
  if (span !== undefined) base.span = span;
  return base;
}
export const mu = [1.399082139134407, 0.14191459026187658];
export const D = [1.6974323936738074, 0.28000224535353485];
export const price = span => raceProbabilities(mu, { D, base: makeBase(span), window: "span", points: 8193 })[0];
"""


@pytestmark_node
def test_typed_span_prices_the_rare_mode_like_a_plain_one():
    got = _node(MIXTURE + """
console.log(JSON.stringify([price([25, 8]), price(new Float64Array([25, 8])), price(new Int32Array([25, 8]))]));
""")
    plain, typed, ints = got
    assert abs(plain - 0.005) < 1e-9                       # the exact rare-mode price
    assert typed == plain and ints == plain                # was 5e-324


@pytestmark_node
@pytest.mark.parametrize("span", ["[25, 8, 3]", "[25]", "[NaN, 8]", "[25, -8]", "[25, Infinity]", '"wide"', "[25, 0]"])
def test_malformed_span_is_refused_not_a_silent_fallback(span):
    got = _node(MIXTURE + f"""
let msg = null;
try {{ price({span}); }} catch (e) {{ msg = String(e.message); }}
console.log(JSON.stringify(msg));
""")
    assert got is not None and "base.span" in got


KINKS = """
import { BASES } from "./docs/js/winning/races.mjs";
import { topKProbabilities, locScaleFromTopkPair } from "./docs/js/winning/topk.mjs";
const mu = [-1.0758897613734006, .8221703173592687, 1.2747228138148785, -1.7811144122853875, .7601110424846411];
const sd = [.5289511696798439, 1.2097101565229562, .9433803641440728, 1.3242591310799288, 1.2509595951816537];
const D = sd.map(x => x * x);
export function laplace(kinks) {
  const f = z => BASES.laplace(z);
  for (const key of ["cdf", "logCdf", "logSf", "resolution", "span"])
    if (BASES.laplace[key] !== undefined) f[key] = BASES.laplace[key];
  if (kinks !== undefined) f.kinks = kinks;
  return f;
}
export function fit(kinks) {
  const base = laplace(kinks);
  const q1 = topKProbabilities(mu, 1, { D, base, points: 513 });
  const q2 = topKProbabilities(mu, 2, { D, base, points: 513 });
  const r = locScaleFromTopkPair(q1, 1, q2, 2, { base, points: 513, nIter: 60, tol: 1e-8, returnInfo: true });
  return { converged: r.info.converged, sd0: r.sd[0], mu: r.mu };
}
"""


@pytestmark_node
def test_typed_kinks_take_the_split_lattice_like_plain_ones():
    got = _node(KINKS + """
console.log(JSON.stringify([fit([0]), fit(new Float64Array([0])), fit(undefined)]));
""")
    plain, typed, smooth = got
    assert plain["converged"] and typed["converged"]
    assert typed["sd0"] == plain["sd0"] and typed["mu"] == plain["mu"]
    # the kinks move the inverse: without them this default grid did not converge (#267)
    assert not smooth["converged"] or abs(smooth["sd0"] - plain["sd0"]) > 1e-6


@pytestmark_node
@pytest.mark.parametrize("kinks", ['[NaN]', '["a"]', '"0"', '[0, Infinity]'])
def test_malformed_kinks_are_refused(kinks):
    got = _node(KINKS + f"""
let msg = null;
try {{ fit({kinks}); }} catch (e) {{ msg = String(e.message); }}
console.log(JSON.stringify(msg));
""")
    assert got is not None and "base.kinks" in got


def _laplace_with(kinks):
    f = lambda z: BASES["laplace"](z)
    for key in ("cdf", "log_cdf", "log_sf", "resolution", "span"):
        if hasattr(BASES["laplace"], key):
            setattr(f, key, getattr(BASES["laplace"], key))
    if kinks is not None:
        f.kinks = kinks
    return f


def test_python_reads_a_numpy_kink_array_as_kinks():
    assert _base_kinks(_laplace_with(np.array([0.0]))) == (0.0,)       # bool(array([0.])) is False
    assert _base_kinks(_laplace_with(np.array([-0.5, 0.5]))) == (-0.5, 0.5)
    assert _base_kinks(_laplace_with((0.0,))) == (0.0,)
    assert _base_kinks(_laplace_with(None)) == ()
    assert _base_kinks(BASES["laplace"]) == (0.0,)
    with pytest.raises(ValueError, match="kinks"):
        _base_kinks(_laplace_with(np.array([0.0, np.nan])))
    mu = np.array([-0.3, 0.1, 0.5, -1.2])
    D = np.array([0.4, 1.1, 0.8, 1.6])
    a = race_probabilities(mu, D=D, base=_laplace_with(np.array([0.0])), kink_split=True, points=257)
    b = race_probabilities(mu, D=D, base=_laplace_with((0.0,)), kink_split=True, points=257)
    np.testing.assert_array_equal(a, b)
