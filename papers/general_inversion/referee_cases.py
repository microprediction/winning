"""Generate seeded referee cases and lattice answers -> cases.json.

Three families:
  central:     the bench 'alt' rank-2 family, n = 10 and 30
  tail:        n = 8, spread mu so the smallest p reaches ~1e-11
  independent: diagonal covariance, n = 12 (admits a 1-D adaptive-quad
               referee in Python to ~1e-12, independent of R)

referee.R replays every probability through mvtnorm (Genz-Bretz) and
TruncatedNormal (Botev, bounded relative error); referee_check.py
compares and runs the invariance battery.
"""
import json

import numpy as np
from winning.factor.races import race_probabilities

rng = np.random.default_rng(4)
cases = []

for n in (10, 30):
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.5 + rng.random(n)
    p = race_probabilities(mu, V=V, D=D, points=513)
    cases.append({"name": f"central_n{n}", "mu": mu.tolist(),
                  "V": V.tolist(), "D": D.tolist(), "p": p.tolist()})

mu = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5])
V = rng.normal(size=(8, 2)) * 0.3
D = 0.6 + 0.4 * rng.random(8)
p = race_probabilities(mu, V=V, D=D, points=2049)
cases.append({"name": "tail_n8", "mu": mu.tolist(), "V": V.tolist(),
              "D": D.tolist(), "p": p.tolist()})

mu = rng.normal(size=12) * 1.5
D = 0.4 + rng.random(12)
p = race_probabilities(mu, V=np.zeros((12, 1)), D=D, points=1025)
cases.append({"name": "indep_n12", "mu": mu.tolist(),
              "V": np.zeros((12, 1)).tolist(), "D": D.tolist(),
              "p": p.tolist()})

json.dump(cases, open("cases.json", "w"))
for c in cases:
    p = np.array(c["p"])
    print(f"{c['name']:12s} n={len(p):3d}  min p = {p.min():.3e}  "
          f"sum = {p.sum():.12f}")
