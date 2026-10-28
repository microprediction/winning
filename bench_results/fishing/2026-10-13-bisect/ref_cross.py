import json, sys, numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import winning.likelihood as L
from winning.likelihood import choice_loglik_and_score
from winning.mnprobit import MNProbit
S = sys.argv[1]
base = json.load(open(f"{S}/py_8e1815f4.json"))
X = np.array(base["X"]); choice = np.array(base["choice"])
m = MNProbit(X, choice, intercepts=True, r=2)
def referee(theta):
    beta, V = m._unpack(np.array(theta)); mu = m.X @ beta; vals = []
    for seed in (101, 102):
        u = qmc.Sobol(3, scramble=True, seed=seed).random(2 ** 15)
        F = ndtri(np.clip(u, 1e-12, 1 - 1e-12)); W = np.full(len(F), 1.0 / len(F))
        orig = L.nodes_for_likelihood
        L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
        try: vals.append(choice_loglik_and_score(mu, V, choice)[0])
        finally: L.nodes_for_likelihood = orig
    return np.mean(vals)
for tag in ("8e1815f4", "b3b55b291d970ac6050b8bc587443a48c28117cf", "216700b500b582c1cb421dde50ba996128eba243", "29acbdc64dd0251304583bdc546ca8c2de6c50e2"):
    th = json.load(open(f"{S}/py_{tag}.json"))["python"]["theta"]
    print(tag[:8], "theta scored by referee at this checkout:", round(referee(th), 2))
