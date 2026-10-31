"""The pair-mixture inverse solves the map the forward exposes (#902).

_pair_mixture_gap solved the raw component sum of the smaller tail while
_pair_mixture, and so race_probabilities, normalize the two tails by their
sum. The maps agree analytically and differ in the last bit; on a sharp
mixture plateau, where the derivative is ~0, one ulp was a root the solver
chased to a gap of 2e48, the public-forward certification refused it, and
the generic sweeps returned a share of 0.067 for a target of 0.246. The
browser shared the defect bit for bit.
"""
import json
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from winning.factor.races import abilities_from_race, race_probabilities

NODE = shutil.which("node")
OPTS = dict(V=[[1], [0]], D=[2e-6, 2e-6], F=[[-1], [0.68], [1]], W=[0.08857, 0.2373, 1])
TRUTH = [-0.415, 0.415]


def test_sharp_mixture_plateau_inverts_through_the_scalar_path():
    target = race_probabilities(TRUTH, **OPTS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu, info = abilities_from_race(target, **OPTS, return_info=True)
    assert info["converged"] and info["iterations"] == 0          # was 60 sweeps, unconverged
    assert np.isfinite(mu).all()
    back = race_probabilities(mu, **OPTS)
    assert np.abs(np.log(back) - np.log(target)).max() < 1e-8    # was 1.30


@pytest.mark.parametrize("seed", range(6))
def test_random_three_atom_laws_round_trip(seed):
    rng = np.random.default_rng(seed)
    F = np.sort(rng.uniform(-1.5, 1.5, 3)).reshape(3, 1)
    W = rng.uniform(0.05, 1.0, 3)
    opts = dict(V=[[1.0], [0.0]], D=[rng.uniform(1e-6, 1e-2)] * 2, F=F, W=W)
    truth = [-0.3, 0.3]
    target = race_probabilities(truth, **opts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, info = abilities_from_race(target, **opts, return_info=True)
    assert info["converged"], info
    back = race_probabilities(mu, **opts)
    assert np.abs(np.log(back) - np.log(target)).max() < 1e-8


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_inverts_the_fixture_identically():
    script = f"""
import {{ raceProbabilities, abilitiesFromRace }} from "./docs/js/winning/races.mjs";
console.warn = () => {{ throw new Error("warned"); }};
const opts = {json.dumps(OPTS)};
const target = raceProbabilities({json.dumps(TRUTH)}, opts);
const fit = abilitiesFromRace(target, {{ ...opts, returnInfo: true }});
const back = raceProbabilities(fit.mu, opts);
console.log(JSON.stringify({{ fit, resid: Math.max(...back.map((x, i) => Math.abs(Math.log(x) - Math.log(target[i])))) }}));
"""
    got = json.loads(subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True,
                                    text=True, check=True).stdout.strip().splitlines()[-1])
    assert got["fit"]["converged"] and got["fit"]["iterations"] == 0
    assert got["resid"] < 1e-8
    target = race_probabilities(TRUTH, **OPTS)
    mu, _ = abilities_from_race(target, **OPTS, return_info=True)
    # on the plateau a range of gaps reprices the target exactly, so the
    # two ports agree to where their last Newton steps landed, not bitwise
    np.testing.assert_allclose(got["fit"]["mu"], mu, rtol=0, atol=1e-6)
