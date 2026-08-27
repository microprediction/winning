"""Generate cross-language parity vectors from the python reference.

The skaters pattern: a registry of named scenarios runs on the reference
implementation (pure python path, WINNING_PURE=1 recommended) and dumps
inputs AND outputs into parity/vectors.json. Every port (R: check.R;
rust exercises the same numbers through tests/test_rust_parity.py)
rebuilds the same-named scenarios from the embedded inputs and asserts
agreement. Inputs are embedded so every language consumes identical
floats.

Run:  WINNING_PURE=1 python parity/gen_vectors.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   block_race_jacobian,
                                   nested_race_jacobian,
                                   tree_race_jacobian,
                                   abilities_from_block_race)
from winning.factor.structures import Tree
from winning.factor.polish import race_jacobian, polish_race
from winning.lattice import skew_normal_density, state_prices_from_offsets
from winning.lattice_calibration import dividend_implied_ability

TOL_DEFAULT = 1e-10


def _linkage_Z(n):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    blocks = np.repeat(np.arange(4), n // 4)
    superb = blocks // 2
    R = (0.15 + 0.25 * (superb[:, None] == superb[None, :])
         + 0.35 * (blocks[:, None] == blocks[None, :]))
    np.fill_diagonal(R, 1.0)
    Z = linkage(squareform(np.sqrt(0.5 * (1.0 - R)), checks=False),
                method="average")
    return np.round(Z, 12).tolist()


def make_inputs(seed=2026):
    rng = np.random.default_rng(seed)
    n = 12
    mu = np.round(rng.normal(size=n), 12)
    V1 = np.round(rng.normal(size=(n, 1)) * 0.4, 12)
    V2 = np.round(rng.normal(size=(n, 2)) * 0.35, 12)
    D = np.round(0.4 + rng.random(n), 12)
    cluster = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    loading = np.round(0.2 + 0.3 * rng.random(n), 12)
    loading2 = np.round(rng.normal(size=(n, 2)) * 0.3, 12)
    coupling = np.round(0.15 + 0.2 * rng.random(n), 12)
    parent = [4, 4, 5, 5, 6, 6, -1]          # 0-based, -1 = root
    strength = [0.4, 0.3, 0.5, 0.35, 0.25, 0.2, 0.0]
    p_target = rng.dirichlet(np.ones(n))
    p_target = np.round(p_target / p_target.sum(), 12)
    p_target = (p_target / p_target.sum()).tolist()
    return {
        "n": n, "mu": mu.tolist(), "V1": V1.tolist(), "V2": V2.tolist(),
        "D": D.tolist(), "cluster": cluster, "loading": loading.tolist(),
        "loading2": loading2.tolist(), "coupling": coupling.tolist(),
        "parent": parent, "strength": strength, "p_target": p_target,
        "dividends": [2.0, 3.5, 6.0, 12.0, 20.0, 41.0],
        "linkage_Z": _linkage_Z(n),
        "classic_L": 500, "classic_unit": 0.01, "classic_a": 1.5,
    }


def build(inputs):
    mu = np.asarray(inputs["mu"])
    V1 = np.asarray(inputs["V1"])
    V2 = np.asarray(inputs["V2"])
    D = np.asarray(inputs["D"])
    cl = np.asarray(inputs["cluster"])
    ld = np.asarray(inputs["loading"])
    ld2 = np.asarray(inputs["loading2"])
    cp = np.asarray(inputs["coupling"])
    pa = np.asarray(inputs["parent"])
    stg = np.asarray(inputs["strength"])
    pt = np.asarray(inputs["p_target"])

    out = {}

    def sc(name, value, tol=TOL_DEFAULT):
        out[name] = {"value": np.asarray(value).tolist(), "tol": tol}

    sc("independent_normal", race_probabilities(mu, D=D, points=257))
    sc("factor1_normal", race_probabilities(mu, V=V1, D=D, points=257))
    sc("factor2_normal", race_probabilities(mu, V=V2, D=D, points=257))
    p, sl = race_probabilities(mu, V=V2, D=D, points=257, return_slopes=True)
    sc("factor2_slopes", sl)
    sc("factor2_span", race_probabilities(mu, V=V2, D=D, points=501,
                                          window="span"))
    sc("gumbel_independent", race_probabilities(
        mu, D=np.full(len(mu), np.pi ** 2 / 6.0), base="gumbel",
        points=1001))
    sc("blocks_r1", block_race_probabilities(mu, cl, ld, D, points=257))
    sc("blocks_r2", block_race_probabilities(mu, cl, ld2, D, points=257))
    sc("nested", nested_race_probabilities(mu, cl, ld, D, coupling=cp,
                                           gamma=0.7, points=257))
    sc("tree", tree_race_probabilities(mu, cl, ld, D, pa, stg, points=257))
    sc("jacobian_factor", race_jacobian(mu, V=V1, D=D, points=257), 1e-9)
    sc("jacobian_blocks", block_race_jacobian(mu, cl, ld, D, points=257),
       1e-9)
    sc("jacobian_nested", nested_race_jacobian(mu, cl, ld, D, coupling=cp,
                                               gamma=0.7, points=257), 1e-9)
    sc("invert_factor", abilities_from_race(pt, V=V1, D=D, points=257),
       1e-7)
    mub, res, _ = abilities_from_block_race(pt, cl, ld, D, points=257)
    sc("invert_blocks", mub, 1e-7)

    density = skew_normal_density(L=inputs["classic_L"],
                                  unit=inputs["classic_unit"],
                                  a=inputs["classic_a"])
    ability = dividend_implied_ability(inputs["dividends"], density)
    sc("classic_ability", np.asarray(ability, float), 1e-8)
    sc("classic_state_prices",
       state_prices_from_offsets(density, [float(a) for a in ability]))

    pp, mup, info = polish_race(p0=pt, V=V1, D=D, points=257,
                                name_caps=0.15)
    sc("polish_p", pp, 5e-4)
    sc("polish_mu", mup, 5e-3)

    sc("jacobian_tree", tree_race_jacobian(mu, cl, ld, D, pa, stg,
                                           points=257), 1e-9)
    tree = Tree.from_linkage(np.asarray(inputs["linkage_Z"]))
    sc("coph_tree", tree_race_probabilities(
        np.asarray(mu), tree.cluster, tree.loading, tree.D,
        tree.parent, tree.strength, points=257))
    ppt, _, _ = polish_race(p0=pt, structure=tree, points=257,
                            name_caps=0.14)
    sc("polish_tree_p", ppt, 5e-4)
    return out


def main():
    inputs = make_inputs()
    vectors = {"inputs": inputs, "scenarios": build(inputs)}
    path = os.path.join(os.path.dirname(__file__), "vectors.json")
    with open(path, "w") as f:
        json.dump(vectors, f)
    print(f"wrote {path}: {len(vectors['scenarios'])} scenarios")


if __name__ == "__main__":
    main()
