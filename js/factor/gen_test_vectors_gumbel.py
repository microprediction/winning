"""Gumbel-base parity vectors for the JavaScript port, generated from the
general race engine (winning.factor.races). Seeded; commit the JSON."""

import json
import sys

import numpy as np

sys.path.insert(0, "../..")
from winning.factor.core import hermite_nodes  # noqa: E402
from winning.factor.races import (  # noqa: E402
    abilities_from_race,
    race_probabilities,
)

rng = np.random.default_rng(2026)
N, K = 12, 2
mu = rng.normal(0, 1.0, N); mu -= mu.mean()
V = rng.normal(0, 0.4, (N, K))
D = rng.uniform(0.5, 1.5, N)
F, W = hermite_nodes(K)

p = race_probabilities(mu, V=V, D=D, F=F, W=W, base="gumbel")
mu_hat = abilities_from_race(p, V=V, D=D, F=F, W=W, base="gumbel")

# independent gumbel must equal softmax exactly (Luce), up to quadrature
p0 = race_probabilities(mu, base="gumbel")
c = np.pi / np.sqrt(6.0)
soft = np.exp(-c * mu) / np.exp(-c * mu).sum()
assert np.abs(p0 - soft).max() < 1e-9, np.abs(p0 - soft).max()

out = {
    "problem": {"mu": mu.tolist(), "V": V.tolist(), "D": D.tolist()},
    "hermite": {"F": F.tolist(), "W": W.tolist()},
    "expected": {"p": p.tolist(), "mu_hat": mu_hat.tolist(),
                 "p_independent": p0.tolist()},
}
with open("test_vectors_gumbel.json", "w") as fjson:
    json.dump(out, fjson)
print("wrote test_vectors_gumbel.json;",
      f"roundtrip {np.abs(mu_hat - mu).max():.2e};",
      f"softmax check {np.abs(p0 - soft).max():.2e}")
