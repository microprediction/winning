"""Compare the lattice against the R referees (referee_out.json), the
independent-case adaptive-quadrature referee, and run the invariance
battery on every competitor.

Battery (each entry a property with an exact expected answer):
  two-runner closed form   p_1 = Phi((mu_2-mu_1)/sqrt(d_11+d_22+||v_1-v_2||^2))
  translation invariance   p(mu + c) = p(mu)
  permutation equivariance p(P mu, P V, P D) = P p(mu, V, D)
  symmetry                 equal mu, exchangeable cov -> p_i = 1/n
Run after referee.R:  python referee_check.py
"""
import itertools
import json

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from winning.factor.races import race_probabilities

cases = json.load(open("cases.json"))
ref = json.load(open("referee_out.json"))

print("=== R referees vs lattice ===")
for c in cases:
    p = np.array(c["p"])
    def _f(x):
        return np.array([float("nan") if v in ("NA", None) else float(v)
                         for v in x])
    g = _f(ref[c["name"]]["genz"])
    ge = _f(ref[c["name"]]["genz_err"])
    b = _f(ref[c["name"]]["botev"])
    g = g / g.sum(); b = b / b.sum()
    rel_g = np.abs(p - g) / np.maximum(g, 1e-300)
    rel_b = np.abs(p - b) / np.maximum(b, 1e-300)
    print(f"{c['name']:12s} maxabs(genz) {np.abs(p-g).max():.2e} "
          f"(genz err bound {ge.max():.1e})  "
          f"maxrel(botev) {rel_b.max():.2e}  "
          f"maxrel(botev) tails p<1e-6: "
          f"{rel_b[np.array(c['p']) < 1e-6].max() if (np.array(c['p']) < 1e-6).any() else float('nan'):.2e}")

print("\n=== independent-case adaptive quadrature (1e-12 target) ===")
c = next(c for c in cases if c["name"] == "indep_n12")
mu, D = np.array(c["mu"]), np.array(c["D"])
sd = np.sqrt(D)
p_lat = np.array(c["p"])


def p_exact(i):
    def integrand(x):
        val = norm.pdf(x, mu[i], sd[i])
        for j in range(len(mu)):
            if j != i:
                val *= norm.sf(x, mu[j], sd[j])
        return val
    v, _ = quad(integrand, mu[i] - 12 * sd[i], mu[i] + 12 * sd[i],
                limit=400, epsabs=1e-14, epsrel=1e-12)
    return v


pe = np.array([p_exact(i) for i in range(len(mu))])
pe = pe / pe.sum()
rel = np.abs(p_lat - pe) / pe
print(f"maxrel {rel.max():.2e}   at tails p<1e-6: "
      f"{rel[pe < 1e-6].max():.2e}   min p checked {pe.min():.1e}")

print("\n=== invariance battery ===")
rng = np.random.default_rng(7)
n = 6
mu = rng.normal(size=n)
V = rng.normal(size=(n, 2)) * 0.4
D = 0.5 + rng.random(n)

# two-runner closed form
m2 = np.array([0.3, -0.2]); V2 = rng.normal(size=(2, 2)) * 0.5
D2 = np.array([0.7, 1.1])
p2 = race_probabilities(m2, V=V2, D=D2, points=1025)
s = np.sqrt(D2.sum() + np.sum((V2[0] - V2[1]) ** 2))
exact = norm.cdf((m2[1] - m2[0]) / s)
print(f"two-runner closed form   err {abs(p2[0] - exact):.2e}")

p0 = race_probabilities(mu, V=V, D=D, points=513)
pc = race_probabilities(mu + 3.7, V=V, D=D, points=513)
print(f"translation invariance   err {np.abs(p0 - pc).max():.2e}")

perm = rng.permutation(n)
pp = race_probabilities(mu[perm], V=V[perm], D=D[perm], points=513)
print(f"permutation equivariance err {np.abs(pp - p0[perm]).max():.2e}")

ps = race_probabilities(np.zeros(n), V=np.ones((n, 1)) * 0.5,
                        D=np.ones(n), points=513)
print(f"symmetry (p = 1/n)       err {np.abs(ps - 1 / n).max():.2e}")
