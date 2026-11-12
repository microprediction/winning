"""The paper's "a fixed rule of fifteen nodes per dimension can miss up to
five percent of the total variation", reproduced on the shipped code.

The fuzz battery found it (log_fuzz2000.txt, seeds below) before the
adaptive rule existed; the shipped default no longer has the fault, so this
script forces the old fixed 15-node Gauss-Hermite rule explicitly and sets it
beside the adaptive default, both against the battery's 2^14-point QMC
referee. Run: python fixed_rule_loss.py
"""
import json
import os
import warnings
import numpy as np
from winning.factor.core import hermite_nodes
from winning.factor.races import race_probabilities

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "fixed_rule_loss.jsonl")    # append-only raw results
src = open(os.path.join(HERE, "fuzz_races.py")).read().split("for case in range(N_CASES):")[0]
exec(src.replace("N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 400", "N_CASES = 0"))

print("seed    n  rank  fixed-15 TV   adaptive TV   (vs 2^14 QMC referee)")
for seed in (10700, 11500, 11540, 11920, 11940):
    rng = np.random.default_rng(seed)
    n, mu, D, V = random_case(rng)
    if V is None:
        continue
    Sig = V @ V.T + np.diag(D)
    ref = mc_reference(mu, Sig, seed=seed)
    F, W = hermite_nodes(V.shape[1], Q=15)
    fixed = race_probabilities(mu, V=V, D=D, F=F, W=W, points=257)
    adapt = race_probabilities(mu, V=V, D=D, points=257)
    print(f"{seed}  {n:3d}  {V.shape[1]:4d}  {0.5*np.abs(fixed-ref).sum():.2e}     "
          f"{0.5*np.abs(adapt-ref).sum():.2e}", flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(seed=seed, n=n, rank=V.shape[1], points=257,
                                ref=ref.tolist(), fixed=fixed.tolist(),
                                adaptive=adapt.tolist())) + "\n")
