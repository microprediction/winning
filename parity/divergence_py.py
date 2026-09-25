"""Emit one ACCEPT/REFUSE verdict per case, for the python port."""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.topk import top_k_probabilities

def num(x):
    if isinstance(x, str):
        return float(x)
    if isinstance(x, list):
        return [num(v) for v in x]
    return x

def run(c):
    mu = num(c.get("mu")); D = num(c.get("D")); V = num(c.get("V"))
    if c["verb"] == "race":
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
