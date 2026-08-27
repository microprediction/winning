"""Randomized three-way differential: sample fresh cases across the
grammars, dump python outputs; check_r.R replays them in the R port."""
import json
import sys
import numpy as np
from winning.factor.races import race_probabilities
from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   block_race_jacobian)

N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
cases = []
for k in range(N_CASES):
    rng = np.random.default_rng(50_000 + k)
    n = int(rng.integers(3, 15))
    mu = rng.normal(size=n)
    D = 0.4 + rng.random(n)
    kind = ["factor", "blocks", "nested", "tree"][k % 4]
    case = {"seed": 50_000 + k, "kind": kind, "mu": mu.tolist(),
            "D": D.tolist()}
    if kind == "factor":
        r = int(rng.integers(1, 3))
        V = rng.normal(size=(n, r)) * rng.uniform(0.1, 0.8)
        case["V"] = V.tolist()
        out = race_probabilities(mu, V=V, D=D, points=257)
    else:
        nc = int(rng.integers(1, max(2, n // 2) + 1))
        cluster = rng.integers(0, nc, size=n)
        ld = rng.uniform(0.05, 0.7, size=n)
        case["cluster"] = cluster.tolist()
        case["loading"] = ld.tolist()
        if kind == "blocks":
            out = block_race_probabilities(mu, cluster, ld, D, points=257)
        elif kind == "nested":
            cp = rng.uniform(0.1, 0.4, size=n)
            g = float(rng.uniform(0.2, 1.0))
            case["coupling"] = cp.tolist(); case["gamma"] = g
            out = nested_race_probabilities(mu, cluster, ld, D, coupling=cp,
                                            gamma=g, points=257)
        else:
            ncl = len(np.unique(cluster))
            parent = np.full(ncl + 1, ncl); parent[-1] = -1
            lam = np.append(rng.uniform(0, 0.6, size=ncl), 0.0)
            # small random hierarchy over clusters when possible
            case["parent"] = parent.tolist(); case["strength"] = lam.tolist()
            out = tree_race_probabilities(mu, cluster, ld, D, parent, lam,
                                          points=257)
    case["p"] = np.asarray(out).tolist()
    cases.append(case)
json.dump(cases, open("r_cases.json", "w"))
print(f"wrote {len(cases)} cases")
