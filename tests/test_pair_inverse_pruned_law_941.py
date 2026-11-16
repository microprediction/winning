"""The pair inverse solves and certifies the factor law the forward prices (#941).

race_probabilities passes a caller's F, W through as_factor_law, which drops
the lightest nodes of combined relative mass <= 1e-15. The caller-node pair
inverse read the raw F, W, so it solved and certified a law with a node the
forward never sees: converged=True at residual 0, while the public reprice of
a 1e-12 share missed by 1e-3 in log. The browser shared the split.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from winning.factor.races import abilities_from_race, race_probabilities

NODE = shutil.which("node")
OPTS = dict(V=[[0.5], [-0.5]], D=[0.5, 0.5], F=[[-10.0], [0.0]], W=[1e-15, 1.0])
G = float(ndtri(1e-12))
TRUTH = [-0.5 * G, 0.5 * G]
TOL = 1e-8


def _reprice(mu, opts, target):
    back = race_probabilities(mu, **opts)
    return float(np.abs(np.log(back) - np.log(target)).max())


def test_negligible_node_fixture_is_certified_by_the_public_forward():
    target = race_probabilities(TRUTH, **OPTS)
    mu, info = abilities_from_race(target, **OPTS, return_info=True)
    resid = _reprice(mu, OPTS, target)
    assert resid < TOL                                # was 9.99e-4
    # the diagnostic is the public forward's own verdict
    assert info["converged"] == (resid < TOL)
    assert info["max_log_residual"] < TOL


# ordinary caller-node pairs, every node non-negligible: the inverse must
# reprice through the public forward exactly as before
ORDINARY = [
    dict(V=[[0.5], [-0.5]], D=[0.5, 0.5], F=[[-1.0], [1.0]], W=[0.5, 0.5]),
    dict(V=[[1.0], [0.0]], D=[0.3, 0.6], F=[[-1.0], [0.2], [1.5]], W=[0.2, 0.5, 0.3]),
    dict(V=[[0.7, 0.1], [-0.2, 0.4]], D=[1.0, 1.0],
         F=[[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]], W=[1.0, 2.0, 1.0]),
    # a light but non-negligible node (1e-10 > 1e-15) is still priced
    dict(V=[[0.5], [-0.5]], D=[0.5, 0.5], F=[[-10.0], [0.0]], W=[1e-10, 1.0]),
]


@pytest.mark.parametrize("opts", ORDINARY)
@pytest.mark.parametrize("truth", [[-0.3, 0.3], [0.8, -0.8], TRUTH])
def test_ordinary_caller_node_pairs_round_trip(opts, truth):
    target = race_probabilities(truth, **opts)
    mu, info = abilities_from_race(target, **opts, return_info=True)
    resid = _reprice(mu, opts, target)
    assert info["converged"] and resid < TOL


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_negligible_node_fixture_is_certified_by_the_public_forward():
    script = f"""
import {{ raceProbabilities, abilitiesFromRace }} from "./docs/js/winning/races.mjs";
const opts = {json.dumps(OPTS)};
const target = raceProbabilities({json.dumps(TRUTH)}, opts);
const fit = abilitiesFromRace(target, {{ ...opts, returnInfo: true }});
const back = raceProbabilities(fit.mu, opts);
console.log(JSON.stringify({{ fit, resid: Math.max(...back.map((x, i) => Math.abs(Math.log(x) - Math.log(target[i])))) }}));
"""
    got = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                                    text=True, check=True).stdout.strip().splitlines()[-1])
    assert got["resid"] < TOL                         # was 9.99e-4
    assert got["fit"]["converged"] == (got["resid"] < TOL)
    assert got["fit"]["maxLogResidual"] < TOL


TIED_NODES = np.array([[-20.0], [0.0], [20.0]])
TIED_WEIGHTS = np.array([8e-16, 1.0, 8e-16])
RARE_GAP = float(ndtri(1e-16))
RARE_MU = [-0.5 * RARE_GAP, 0.5 * RARE_GAP]


def _tied_node_target():
    # Independent oracle: integrate the two Gaussian tails over all
    # three nodes. The two tied atoms exceed the pruning budget together.
    z = RARE_GAP - TIED_NODES[:, 0]
    return np.array([TIED_WEIGHTS @ ndtr(z), TIED_WEIGHTS @ ndtr(-z)]) / TIED_WEIGHTS.sum()


def test_tied_nodes_at_pruning_cutoff_are_order_invariant():
    target = _tied_node_target()
    for nodes in (TIED_NODES, TIED_NODES[::-1]):
        opts = dict(OPTS, F=nodes, W=TIED_WEIGHTS)
        assert _reprice(RARE_MU, opts, target) < TOL
        mu, info = abilities_from_race(target, **opts, return_info=True)
        assert info["converged"] and _reprice(mu, opts, target) < TOL


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_keeps_tied_nodes_together_at_pruning_cutoff():
    target = _tied_node_target()
    script = f"""
import {{ raceProbabilities, abilitiesFromRace }} from "./docs/js/winning/races.mjs";
import {{ asFactorLaw }} from "./js/factor/factor_race.mjs";
const F = {json.dumps(TIED_NODES.tolist())}, W = {json.dumps(TIED_WEIGHTS.tolist())};
const mu = {json.dumps(RARE_MU)}, target = {json.dumps(target.tolist())};
const miss = p => Math.max(...p.map((x, i) => Math.abs(Math.log(x) - Math.log(target[i]))));
for (const nodes of [F, F.slice().reverse()]) {{
  const opts = {{ V: [[.5], [-.5]], D: [.5, .5], F: nodes, W }};
  const fit = abilitiesFromRace(target, {{ ...opts, returnInfo: true }});
  if (!fit.converged || miss(raceProbabilities(mu, opts)) >= {TOL} ||
      miss(raceProbabilities(fit.mu, opts)) >= {TOL} || asFactorLaw(nodes, W).F.length !== 3)
    throw new Error('Factor-law pruning split equal weights');
}}
"""
    subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                   text=True, check=True)
