import json, sys, numpy as np
import winning.likelihood as L
from winning.mnprobit import MNProbit
inp = json.load(open(sys.argv[1]))
m = MNProbit(np.array(inp["X"]), np.array(inp["choice"]), intercepts=True, r=2)
F, W = L.nodes_for_likelihood(2, 151, 151, 0.0)
for f in sys.argv[2:]:
    th = np.array([float(v) for v in open(f).read().strip().split(",")])
    beta, V = m._unpack(th); mu = m.X @ beta
    orig = L.nodes_for_likelihood
    L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
    try: truth = L.choice_loglik_and_score(mu, V, m.choice)[0]
    finally: L.nodes_for_likelihood = orig
    print(f, "sharpness %.2f truth(GH151) %.2f" % (L.sharpness_bound(V, np.ones(m.J)), truth))
