"""Experiment 37: the photo-finish weight is the differentiated racing
multiplicity.

Cotton (2021) tracks a scalar "multiplicity" on the lattice: the expected
number of runners sharing the winning cell, used to split dead-heat
payoffs (a k-way tie pays each winner 1/k). This paper's weight
w_ij = int g_i g_j prod_{l!=i,j} F_l dx is the continuum, pairwise density
of the same event and equals -dp_i/dmu_j.

Verified here (independent max-wins Gaussian race, N=6, seed 7):
  1. -dp_i/dmu_j = w_ij off the diagonal (to ~1e-10);
  2. sum_{i<j} w_ij = (1/2) tr(J) exactly;
  3. the racing multiplicity's excess over one, per unit lattice spacing,
     converges to sum_{i<j} w_ij as dx -> 0: E[#tied at winning cell - 1]/dx
     -> 1/2 tr(J).

Run:  python experiments/exp37_multiplicity/run_multiplicity.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)
N = 6
mu = rng.normal(0, 1.0, N)
sd = rng.uniform(0.7, 1.3, N)

x = np.linspace(mu.min() - 10, mu.max() + 10, 20001)
dxf = x[1] - x[0]
phi = norm.pdf((x[None, :] - mu[:, None]) / sd[:, None]) / sd[:, None]
Phi = np.clip(norm.cdf((x[None, :] - mu[:, None]) / sd[:, None]), 1e-300, 1)
field = np.log(Phi).sum(0)
w = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        w[i, j] = np.sum(phi[i] * phi[j]
                         * np.exp(field - np.log(Phi[i]) - np.log(Phi[j]))) * dxf
sum_w = w.sum()

def p_forward(m):
    z = (x[None, :] - m[:, None]) / sd[:, None]
    ph = norm.pdf(z) / sd[:, None]; PH = np.clip(norm.cdf(z), 1e-300, 1)
    fld = np.log(PH).sum(0)
    return np.array([np.sum(ph[i] * np.exp(fld - np.log(PH[i]))) * dxf
                     for i in range(N)])

eps = 1e-4
J = np.zeros((N, N))
for j in range(N):
    e = np.zeros(N); e[j] = eps
    J[:, j] = (p_forward(mu + e) - p_forward(mu - e)) / (2 * eps)
off = ~np.eye(N, dtype=bool)
jac_err = float(np.abs(-J[off] - (w + w.T)[off]).max())

R = 20_000_000
U = mu[None, :] + sd[None, :] * rng.standard_normal((R, N))
rows = ["quantity,value"]
rows += [f"sum_w,{sum_w:.6f}", f"half_tr_J,{0.5*np.trace(J):.6f}",
         f"jacobian_offdiag_err,{jac_err:.3e}"]
print(f"sum_{{i<j}} w_ij            = {sum_w:.5f}")
print(f"(1/2) tr(J)              = {0.5*np.trace(J):.5f}")
print(f"max |(-dp/dmu) - w| off  = {jac_err:.2e}")
print(f"\n{'dx':>6} {'excess':>10} {'excess/dx':>10} {'target sum_w':>12}")
for dxl in (0.2, 0.1, 0.05, 0.025):
    cell = np.floor(U / dxl)
    top = cell.max(axis=1, keepdims=True)
    excess = ((cell == top).sum(axis=1) - 1).mean()
    print(f"{dxl:6.3f} {excess:10.6f} {excess/dxl:10.4f} {sum_w:12.4f}")
    rows.append(f"excess_over_dx_at_dx_{dxl},{excess/dxl:.4f}")
(HERE / "results.csv").write_text("\n".join(rows) + "\n")
print("wrote results.csv")
