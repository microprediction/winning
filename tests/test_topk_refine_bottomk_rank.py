"""Bottom-k and rank marginals share top-k's bounded 2p - 1 refinement.

#800: bottom_k_probabilities threw "membership 1.98" at the default 513
points on a valid Laplace field that 1025 resolves. It is the #769 top-k
field with the abilities negated, so the refined answer must agree with
top-k on -mu. #819: rank_probabilities rejected a Laplace field at 513
and 1025 points that 2049 resolves. A call that passes first time
evaluates one lattice, as before."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from winning.factor import topk as T

# #800: five-runner Laplace field, bottom-2
MU_800 = [-3.02, 0.08, 3.70, -4.74, -6.13]
D_800 = [2.53, 15.04, 0.30, 22.94, 2.24]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
MOD = json.dumps(Path(ROOT, "docs", "js", "winning", "topk.mjs").as_uri())


def _node(script):
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True,
                         timeout=300)
    return json.loads(out.stdout)


def _count_lattices(monkeypatch):
    calls = []
    real = T._topk_independent

    def counting(*a, **k):
        calls.append(a[4])
        return real(*a, **k)
    monkeypatch.setattr(T, "_topk_independent", counting)
    return calls


def test_800_bottom_k_default_is_refined():
    q = T.bottom_k_probabilities(MU_800, 2, D=D_800, base="laplace")
    q1025 = T.bottom_k_probabilities(MU_800, 2, D=D_800, base="laplace",
                                     points=1025)
    assert np.array_equal(q, q1025)
    assert abs(q.sum() - 2.0) < 1e-12
    top = T.top_k_probabilities(-np.asarray(MU_800), 2, D=D_800,
                                base="laplace")
    assert np.abs(q - top).max() < 1e-12


def test_800_bottom_k_factor_path_is_refined():
    """The factor mixture takes the same policy: a rank-one loading
    small enough to leave the field as defective as #800's."""
    V = 1e-3 * np.ones((5, 1))
    q = T.bottom_k_probabilities(MU_800, 2, V=V, D=D_800, base="laplace")
    q1025 = T.bottom_k_probabilities(MU_800, 2, V=V, D=D_800,
                                     base="laplace", points=1025)
    assert np.array_equal(q, q1025)


def test_800_unresolved_field_stays_bounded():
    with pytest.raises(RuntimeError, match="refined from points=129 to 513"):
        T.bottom_k_probabilities(MU_800, 2, D=D_800, base="laplace",
                                 points=129)


@pytest.mark.parametrize("mu,k,base", [
    ([0.3, -0.2, 0.9, -1.1, 0.4], 2, "normal"),
    ([0.3, -0.2, 0.9, -1.1, 0.4], 1, "laplace"),
    ([-3.0, 0.1, 3.7, -4.7, -6.1], 2, "normal"),
    ([1.2, -0.5, 0.0, 2.2], 3, "logistic"),
])
def test_800_ordinary_bottom_k_evaluates_one_lattice(monkeypatch, mu, k,
                                                     base):
    calls = _count_lattices(monkeypatch)
    q = T.bottom_k_probabilities(mu, k, base=base)
    assert calls == [513]
    raw = T._topk_independent(np.asarray(mu, float), np.ones(len(mu)), k,
                              T.BASES[base], 513, upper=True)
    assert np.array_equal(q, T._checked_topk(raw, k, "bottom-k race"))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_800_browser_bottom_k_matches_python():
    got = _node(f"""
import {{ bottomKProbabilities }} from {MOD};
const mu = {json.dumps(MU_800)}, D = {json.dumps(D_800)};
const q = bottomKProbabilities(mu, 2, {{ D, base: "laplace" }});
let err = null;
try {{ bottomKProbabilities(mu, 2, {{ D, base: "laplace", points: 129 }}); }}
catch (e) {{ err = e.message; }}
console.log(JSON.stringify({{ q, err }}));
""")
    q = T.bottom_k_probabilities(MU_800, 2, D=D_800, base="laplace")
    assert np.abs(np.asarray(got["q"]) - q).max() < 1e-12
    assert "refined from points=129 to 513" in got["err"]


# #819: rank marginals rejected the same field (abilities as in the
# issue) at 513 and 1025 points; 2049 resolves it, inside the 4x cap.
MU_819 = [3.02, -0.08, -3.70, 4.74, 6.13]


def test_819_rank_default_is_refined():
    P = T.rank_probabilities(MU_819, D=D_800, base="laplace")
    P2049 = T.rank_probabilities(MU_819, D=D_800, base="laplace",
                                 points=2049)
    assert np.array_equal(P, P2049)
    assert np.abs(P.sum(axis=1) - 1).max() < 1e-12
    assert np.abs(P.sum(axis=0) - 1).max() < 5e-3


def test_819_rank_factor_path_is_refined():
    V = 1e-3 * np.ones((5, 1))
    P = T.rank_probabilities(MU_819, D=D_800, base="laplace", V=V)
    P2049 = T.rank_probabilities(MU_819, D=D_800, base="laplace", V=V,
                                 points=2049)
    assert np.array_equal(P, P2049)


def test_819_unresolved_rank_field_stays_bounded():
    with pytest.raises(RuntimeError,
                       match="refined from points=257 to 1025"):
        T.rank_probabilities(MU_819, D=D_800, base="laplace", points=257)


@pytest.mark.parametrize("mu,base", [
    ([0.3, -0.2, 0.9, -1.1, 0.4], "laplace"),
    ([0.3, -0.2, 0.9, -1.1, 0.4], "logistic"),
    ([3.0, -0.1, -3.7, 4.7, 6.1], "logistic"),
    ([1.2, -0.5, 0.0, 2.2], "laplace"),
])
def test_819_ordinary_rank_evaluates_one_lattice(monkeypatch, mu, base):
    calls = []
    real = T._resolved_points

    def counting(lo, hi, sd, points, res=1.0):
        calls.append(points)
        return real(lo, hi, sd, points, res)
    monkeypatch.setattr(T, "_resolved_points", counting)
    T.rank_probabilities(mu, base=base)
    assert calls == [513]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_819_browser_rank_matches_python():
    got = _node(f"""
import {{ rankProbabilities }} from {MOD};
const mu = {json.dumps(MU_819)}, D = {json.dumps(D_800)};
const P = rankProbabilities(mu, {{ D, base: "laplace" }});
let err = null;
try {{ rankProbabilities(mu, {{ D, base: "laplace", points: 257 }}); }}
catch (e) {{ err = e.message; }}
console.log(JSON.stringify({{ P, err }}));
""")
    P = T.rank_probabilities(MU_819, D=D_800, base="laplace")
    assert np.abs(np.asarray(got["P"]) - P).max() < 1e-12
    assert "refined from points=257 to 1025" in got["err"]
