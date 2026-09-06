"""Fixtures for MvNormalCDFFast.jl from the python reference
(winning.fastmvn is the spec). Exact-path cases pinned tight; the
deep-tail cases loose (the reference uses Sobol there, the Julia port
deterministic recentered GH).

Run:  python julia/MvNormalCDFFast/gen_fixtures.py
"""
import json
import os

import numpy as np

from winning.fastmvn import mvn_cdf_fast_info

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

cases = []


def add(name, n, r, lower, upper, tight, seed):
    g = np.random.default_rng(seed)
    V = np.round(g.normal(size=(n, r)) * 0.5, 12)
    D = np.round(0.5 + g.random(n), 12)
    mu = np.round(g.normal(size=n) * 0.3, 12)
    p, method = mvn_cdf_fast_info(lower=lower, upper=upper, mean=mu,
                                  V=V, D=D)
    cases.append({
        "name": name, "V": V.tolist(), "D": D.tolist(), "mu": mu.tolist(),
        "lower": (None if lower is None else list(np.broadcast_to(lower, n).astype(float))),
        "upper": (None if upper is None else list(np.broadcast_to(upper, n).astype(float))),
        "p": p, "method": method, "tight": tight,
    })
    print(f"{name:24s} p={p:.6e} ({method})")


add("rank1_onesided", 8, 1, None, 0.7, True, 1)
add("rank1_rectangle", 8, 1, -1.0, 1.2, True, 2)
add("rank2_onesided", 12, 2, None, 0.5, True, 3)
add("rank2_rectangle", 12, 2, -0.8, 1.5, True, 4)
add("rank2_bigger", 40, 2, None, 0.0, True, 5)
add("rank1_deeptail", 8, 1, None, -3.2, False, 6)
add("rank2_deeptail", 10, 2, None, -3.0, False, 7)

with open(os.path.join(HERE, "test", "vectors.json"), "w") as f:
    json.dump({"cases": cases}, f)
print("wrote vectors.json")
