"""Emit one ACCEPT/REFUSE verdict per case, for the python port."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.topk import (bottom_k_probabilities, rank_probabilities,
                                 top_k_probabilities)
from winning.factor.core import hermite_nodes

def num(x):
    if isinstance(x, str):
        return float(x)
    if isinstance(x, list):
        return [num(v) for v in x]
    return x

def run(c):
    mu = num(c.get("mu")); D = num(c.get("D")); V = num(c.get("V"))
    if c.get("Vt") is not None:
        V = num(c["Vt"])
    if c["verb"] == "hermite":
        F, W = hermite_nodes(c["r"], c["order"])
        # SHAPE, not the nodes: every port must agree on how many nodes
        # in how many columns, which is what k and the order decide
        return ("ACCEPT" if np.isfinite(F).all() and np.isfinite(W).all()
                else "ACCEPT_NONFINITE",
                [float(F.shape[0]), float(F.shape[1])])
    if c["verb"] == "rank":
        p = rank_probabilities(np.asarray(mu, float), D=D)
    elif c["verb"] == "bottomk":
        p = bottom_k_probabilities(np.asarray(mu, float), c["k"], D=D)
    elif c["verb"] == "race":
        p = race_probabilities(np.asarray(mu, float), V=V, D=D)
    elif c["verb"] == "inverse":
        p = abilities_from_race(np.asarray(num(c["p"]), float), D=D)
    else:
        p = top_k_probabilities(np.asarray(mu, float), c["k"], D=D)
    a = np.asarray(p, float)
    return ("ACCEPT" if np.isfinite(a).all() else "ACCEPT_NONFINITE",
            [float(v) for v in a.ravel()[:6]])

out = {}
cases = json.load(open(sys.argv[1]))["cases"]
for c in cases:
    try:
        verdict, val = run(c)
        out[c["id"]] = {"verdict": verdict, "value": val}
    except Exception as e:
        out[c["id"]] = {"verdict": "REFUSE", "error": type(e).__name__}
print(json.dumps(out))
