"""Python against R on the sharp-factor case (Section 7, stress boundaries).

Past sharpness 3 both implementations replace Gauss-Hermite factor nodes
with low-discrepancy nodes: scrambled Sobol in Python, Halton in R. This
script prices the factor-strength case of break.py section H (n = 20,
rank two, loading scale 4) with each implementation's default call and
prints the total variation between the two, and of each against a
2^22-path reference.

Seed: default_rng(99) for the problem, Sobol seed 1 for the reference.
Needs Rscript and the R package jsonlite; the R port is sourced from
r/winning/R in this repository.

    python port_agreement.py
"""
import json
import os
import subprocess
import tempfile

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from winning.factor import race_probabilities

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

rng = np.random.default_rng(99)
n = 20
mu = rng.normal(size=n)
V = rng.normal(size=(n, 2)) * 4.0
D = 0.5 + rng.random(n)

p_py = race_probabilities(mu, V=V, D=D)

R_CODE = r"""
args <- commandArgs(TRUE)
for (f in list.files(file.path(args[1], "r", "winning", "R"), full.names = TRUE)) source(f)
x <- jsonlite::fromJSON(args[2])
p <- race_probabilities(x$mu, V = as.matrix(x$V), D = x$D)
cat(jsonlite::toJSON(as.numeric(p), digits = NA))
"""
with tempfile.TemporaryDirectory() as tmp:
    case = os.path.join(tmp, "case.json")
    code = os.path.join(tmp, "run.R")
    json.dump(dict(mu=mu.tolist(), V=V.tolist(), D=D.tolist()), open(case, "w"))
    open(code, "w").write(R_CODE)
    out = subprocess.run(["Rscript", code, ROOT, case], check=True,
                         capture_output=True, text=True).stdout
p_r = np.array(json.loads(out))

L = np.linalg.cholesky(V @ V.T + np.diag(D))
z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1).random_base2(22),
                  1e-12, 1 - 1e-12)).T
ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0), minlength=n) / z.shape[1]

sharp = np.sqrt(2) * np.max(np.linalg.norm(V, axis=1) / np.sqrt(D))
tv = lambda a, b: 0.5 * np.abs(a - b).sum()
print(f"n={n}, rank 2, loading scale 4, sharpness {sharp:.1f}")
print(f"TV Python default against R default  {tv(p_py, p_r):.2e}")
print(f"TV Python default against reference  {tv(p_py, ref):.2e}")
print(f"TV R default against reference       {tv(p_r, ref):.2e}")
