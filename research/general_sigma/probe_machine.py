"""Which stage of run_ensembles4.py depends on the machine?

For seed 0 of each ensemble it prints a short hash of (a) the generated
correlation matrix, rounded to 1e-10, (b) the Monte Carlo referee's win
counts and (c) the priced probabilities of each arm, rounded to 1e-10,
and the total variation of each arm against the referee. Run it on two
machines in the same pinned environment and compare the lines: the first
column that differs names the stage.

    python probe_machine.py > probe_machine_<name>.txt
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import platform
import warnings

import numpy as np

warnings.simplefilter("ignore")
src = open("run_ensembles4.py").read()
ns = {"__name__": "probe"}
exec(src[:src.index("import csv\noutf")], ns)
n, mu, b, build, M = ns["n"], ns["mu"], ns["b"], ns["build"], ns["M"]


def h(a, digits=10):
    return hashlib.md5(np.round(np.asarray(a, float), digits).tobytes()).hexdigest()[:8]


print(platform.platform(), platform.processor(), "numpy", np.__version__)
np.show_config()
print(f"{'ensemble':22s} C-hash    MC-hash   raw-hash  proj-hash  TV raw      TV projected")
for tag, gen, kw, m in ns["CASES"]:
    seed = 0
    rr = np.random.default_rng(1000 + seed)
    try:
        C = np.asarray(gen(n=n, rng=1000 + seed, **kw))
    except TypeError:
        try:
            C = np.asarray(gen(n, rr))
        except TypeError:
            np.random.seed(1000 + seed)
            C = np.asarray(gen(n=n, **kw))
    pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(seed))
    pr, pp = build(C, "raw", m=m), build(C, "projected", m=m)
    print(f"{tag:22s} {h(C)}  {h(counts, 0)}  {h(pr)}  {h(pp)}   "
          f"{0.5 * np.abs(pr - pmc).sum():.6e}  {0.5 * np.abs(pp - pmc).sum():.6e}",
          flush=True)
