"""Refinement of the Claim 1 check: the first pass showed 2.7-4.4
percent gaps at z ~ -5, which is method error on both sides, not noise.

MC side: occupation bias is ~O(eps); run two bands and extrapolate to
eps -> 0, with dt small enough that the per-step move of Y is well
inside the narrow band.
Lattice side: double the horizon grid and check the integral is stable.
"""
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from run_localtime import lattice_integral, mc_local_time  # noqa: E402

sigma = 1.0
T = 0.5
mu = np.array([-0.3, -0.1, 0.0, 0.25, 0.5])

out = {}
for (i, j), tag in (((0, 1), "pair01"), ((0, 3), "pair03")):
    l48, _, _ = lattice_integral(mu, sigma, T, i, j, n_s=48)
    l96, _, _ = lattice_integral(mu, sigma, T, i, j, n_s=96)
    e_hi, se_hi = mc_local_time(mu, sigma, T, i, j, paths=60_000,
                                dt=2.5e-5, eps=0.08, seed=3)
    e_lo, se_lo = mc_local_time(mu, sigma, T, i, j, paths=60_000,
                                dt=2.5e-5, eps=0.04, seed=4)
    # linear-in-eps extrapolation to zero band width
    mc0 = e_lo + (e_lo - e_hi) * 0.04 / (0.08 - 0.04)
    se0 = np.sqrt((2 * se_lo) ** 2 + se_hi ** 2)
    z = (mc0 - l96) / se0
    print(f"{tag}: lattice 48/96 = {l48:.5f}/{l96:.5f} | "
          f"MC eps .08/.04 = {e_hi:.5f}/{e_lo:.5f} -> eps->0 {mc0:.5f} "
          f"+/- {se0:.5f}  (z = {z:+.2f})")
    out[tag] = dict(lattice_ns48=l48, lattice_ns96=l96,
                    mc_eps08=e_hi, mc_eps04=e_lo, mc_extrap=mc0,
                    se=float(se0), z=float(z))

with open(os.path.join(os.path.dirname(__file__), "results_refine.json"),
          "w") as f:
    json.dump(out, f, indent=2)
print("wrote results_refine.json")
