"""Deletion-row parity vectors for the JavaScript port (#785): python's
safe single-removal ensemble, winning.factor.races.removal_shares, on a
battery of ordinary fields -- normal, gumbel and laplace bases, 3 to 12
runners, sd ratios 1 to 20, with and without a factor. JavaScript's
winProbabilitiesFactor(..., {deletions: true}) must reproduce every row.
Seeded; commit the JSON."""

import json
import sys

import numpy as np

sys.path.insert(0, "../..")
from winning.factor.core import hermite_nodes  # noqa: E402
from winning.factor.races import removal_shares  # noqa: E402

rng = np.random.default_rng(785)
cases = []
for base in ("normal", "gumbel", "laplace"):
    for n in (3, 5, 8, 12):
        for ratio in (1.0, 2.0, 5.0, 20.0):
            for factor in (False, True):
                mu = rng.normal(0, 1.0, n)
                # sds spread geometrically over [smax / ratio, smax]
                smax = rng.uniform(0.5, 1.5)
                sd = smax * ratio ** -rng.uniform(0, 1, n)
                sd[0], sd[-1] = smax, smax / ratio
                D = sd ** 2
                if factor:
                    V = rng.normal(0, 0.4, (n, 1))
                    F, W = hermite_nodes(1)
                else:
                    V = np.zeros((n, 0))
                    F, W = np.zeros((1, 0)), np.ones(1)
                try:
                    q = removal_shares(mu, V=V if factor else None, D=D,
                                       F=F if factor else None,
                                       W=W if factor else None,
                                       base=base).tolist()
                except FloatingPointError:
                    # python's row-mass guard (mass_tol 1e-3) is tighter
                    # than javascript's 5e-3; such a field has no reference
                    q = None
                    print(f"refused: {base} n={n} ratio={ratio} "
                          f"factor={factor}")
                cases.append({
                    "base": base, "mu": mu.tolist(), "V": V.tolist(),
                    "D": D.tolist(), "F": F.tolist(), "W": W.tolist(),
                    "deletions": q,
                })

with open("test_vectors_removal.json", "w") as fjson:
    json.dump({"cases": cases}, fjson)
print(f"wrote test_vectors_removal.json; {len(cases)} cases")
