"""A rank-3+ nested coupling is priced in a basis fixed by its covariance (#911).

The global factor is one N(0, I) draw, so a column sign flip, an
orthogonal rotation and a common shock added to every row of the coupling
are the same race. Python fed the caller's coordinates to the fixed
scrambled-Sobol cloud, so flipping one column of a rank-three coupling
moved a winner share by 12 bp while the browser, which canonicalizes the
coupling first, was bit-identical. The direct forward, the Jacobian and
the structured Nested route now canonicalize as the browser does.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest

from winning.factor.blocks import (_canonical_coupling, nested_race_jacobian,
                                   nested_race_probabilities)
from winning.factor.races import race_probabilities
from winning.factor.structures import Nested

NODE = shutil.which("node")

# the issue's fixture: three singleton clusters, no block loading
MU = np.array([-0.8, 0.1, 0.7])
D = np.array([0.2, 0.15, 0.1])
G3 = np.array([[3.0, -1.0, 2.0], [-2.0, 2.5, 0.5], [1.0, 2.0, -3.0]])


def _generic(n, r, seed):
    rng = np.random.default_rng(seed)
    mu = 0.6 * rng.standard_normal(n)
    G = 0.5 * rng.standard_normal((n, r))
    D = 0.2 + 0.3 * rng.random(n)
    return mu, list(range(n)), [0.0] * n, D, G


def _rotation(r, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((r, r)))
    return Q


def test_issue_fixture_is_invariant_to_a_column_sign_flip():
    kw = dict(points=65, qf=15)
    p = nested_race_probabilities(MU, np.arange(3), np.zeros(3), D, coupling=G3, **kw)
    q = nested_race_probabilities(MU, np.arange(3), np.zeros(3), D,
                                  coupling=G3 * np.array([-1.0, 1.0, 1.0]), **kw)
    assert np.array_equal(p, q)                                   # was 0.00121 apart
    # the coupling moves the race: the invariance is not an independent race in disguise
    indep = nested_race_probabilities(MU, np.arange(3), np.zeros(3), D, coupling=None, **kw)
    assert np.abs(p - indep).max() > 1e-2


@pytest.mark.parametrize("n,r,seed", [(8, 3, 0), (10, 4, 1), (7, 5, 2)])
def test_sign_flip_rotation_and_common_shock_are_the_same_race(n, r, seed):
    mu, cl, ld, Dn, G = _generic(n, r, seed)
    kw = dict(points=65, qf=9)
    p = nested_race_probabilities(mu, cl, ld, Dn, coupling=G, **kw)
    J = nested_race_jacobian(mu, cl, ld, Dn, coupling=G, **kw)
    assert np.abs(p - nested_race_probabilities(mu, cl, ld, Dn, coupling=None, **kw)).max() > 1e-3
    Q = _rotation(r, seed)
    flipped = G * np.where(np.arange(r) == 1, -1.0, 1.0)
    shocked = G + np.full((n, 1), 1.0) @ np.arange(1, r + 1).reshape(1, r)
    for other in (flipped, G @ Q, shocked, G[:, ::-1]):
        q = nested_race_probabilities(mu, cl, ld, Dn, coupling=other, **kw)
        assert np.abs(p - q).max() < 1e-12, np.abs(p - q).max()
        Jq = nested_race_jacobian(mu, cl, ld, Dn, coupling=other, **kw)
        assert np.abs(J - Jq).max() < 1e-12


def test_structured_route_follows_the_direct_one():
    mu, cl, ld, Dn, G = _generic(8, 3, 5)
    Q = _rotation(3, 5)
    a = race_probabilities(mu, structure=Nested(cl, ld, Dn, G, 1.0), points=65)
    b = race_probabilities(mu, structure=Nested(cl, ld, Dn, G @ Q, 1.0), points=65)
    assert np.abs(a - b).max() < 1e-12
    np.testing.assert_allclose(a, nested_race_probabilities(mu, cl, ld, Dn, coupling=G, points=65))


def test_canonical_coupling_drops_null_directions_and_leaves_low_rank_alone():
    # three runners: a centred coupling has rank at most two, so the
    # Gauss-Hermite rules take over and rank is what the race has, not
    # what the caller spelled
    C = _canonical_coupling(G3)
    assert C.shape == (3, 2)
    assert np.abs(C @ C.T - _centred_gram(G3)).max() < 1e-12
    # a rank-one coupling spelled with three columns
    g = np.outer([1.0, -2.0, 0.5, 3.0], [1.0, 2.0, 3.0])
    assert _canonical_coupling(g).shape == (4, 1)
    # all common: the race is independent, one zero column
    assert np.array_equal(_canonical_coupling(np.ones((4, 3))), np.zeros((4, 1)))
    # rank one and two are returned as given
    for g in (np.arange(8.0).reshape(4, 2), np.arange(4.0).reshape(4, 1)):
        assert _canonical_coupling(g) is g


def _centred_gram(g):
    gc = g - g.mean(axis=0)
    return gc @ gc.T


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_and_python_agree_on_the_issue_fixture():
    # after centring the three-runner coupling has rank two and both ports
    # take the tensor Gauss-Hermite rule in the same canonical axes
    script = f"""
import {{ nestedRaceProbabilities }} from "./docs/js/winning/blocks.mjs";
const G = {json.dumps(G3.tolist())};
const flip = G.map(r => [-r[0], r[1], r[2]]);
const kw = {{ qf: 15, points: 65 }};
console.log(JSON.stringify([
  nestedRaceProbabilities({json.dumps(MU.tolist())}, [0, 1, 2], [0, 0, 0], {json.dumps(D.tolist())}, {{ coupling: G, ...kw }}),
  nestedRaceProbabilities({json.dumps(MU.tolist())}, [0, 1, 2], [0, 0, 0], {json.dumps(D.tolist())}, {{ coupling: flip, ...kw }}),
]));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    js, js_flip = json.loads(out.stdout.strip().splitlines()[-1])
    p = nested_race_probabilities(MU, np.arange(3), np.zeros(3), D, coupling=G3, points=65, qf=15)
    np.testing.assert_allclose(js, p, atol=1e-12)
    np.testing.assert_allclose(js_flip, p, atol=1e-12)
