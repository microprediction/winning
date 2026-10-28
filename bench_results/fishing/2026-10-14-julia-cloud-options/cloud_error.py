import json, sys, numpy as np
from scipy.special import ndtri
import winning.likelihood as L
from winning.mnprobit import MNProbit
B = sys.argv[1]
def halton(r, n, off=20):
    primes = [2, 3, 5, 7, 11][:r]; F = np.zeros((n, r))
    for c, b in enumerate(primes):
        for row in range(n):
            i, f, h = row + 1 + off, 1.0 / b, 0.0
            while i > 0:
                h += f * (i % b); i //= b; f /= b
            F[row, c] = ndtri(min(max(h, 1e-12), 1 - 1e-12))
    return F, np.full(n, 1.0 / n)
def rsob(n, s):
    F = np.loadtxt(f"rsob_{n}_{s}.csv", delimiter=",", skiprows=1); return F, np.full(n, 1.0 / n)
base = json.load(open(f"{B}/py_main.json"))
m = MNProbit(np.array(base["X"]), np.array(base["choice"]), intercepts=True, r=2)
pts = [("main py", json.load(open(f"{B}/py_main.json"))["python"]["theta"]),
       ("br jl", json.load(open(f"{B}/jl_branch.json"))["theta_exact"]),
       ("br py", json.load(open(f"{B}/py_branch.json"))["python"]["theta"])]
clouds = {"GH151": L.nodes_for_likelihood(2, 151, 151, 0.0)}
for k in (10, 11, 12, 13): clouds[f"halton2^{k}"] = halton(3, 2**k)
for k in (10, 11, 12):
    for s in range(5): clouds[f"Rsobol2^{k}s{s}"] = rsob(2**k, s)
for k in (10, 12):
    for s in range(5): clouds[f"pysobol2^{k}s{s}"] = L.sobol_normal(2, 1, 2**k, s)
res = {}
for name, th in pts:
    beta, V = m._unpack(np.array(th)); mu = m.X @ beta
    for cn, (F, W) in clouds.items():
        orig = L.nodes_for_likelihood
        L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
        try: res[(name, cn)] = L.choice_loglik_and_score(mu, V, m.choice)[0]
        finally: L.nodes_for_likelihood = orig
for cn in clouds:
    errs = [res[(p, cn)] - res[(p, "GH151")] for p, _ in pts]
    print(f"{cn:16s} " + " ".join(f"{e:+7.2f}" for e in errs) + f"   rms {np.sqrt(np.mean(np.square(errs))):.2f}")
