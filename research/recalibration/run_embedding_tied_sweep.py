"""Embedding-tied (V, D) at scale: K = 100 .. 1000, one sweep.

Supersedes run_embedding_tied_scale.py, which produced a single K = 200 row in
hours. Three changes, each measured rather than assumed:

  nodes     the rank-2 factor was integrated by a 15 x 15 GH product (225
            nodes) and the conditioning dimension by 15 more. Scrambled Sobol
            over the factor plane needs ~64 for the same accuracy (the same
            finding as research/qpo), and the z dimension converges by 9.
            3,375 -> 576 evaluation points, a 5.9x cut.
  dtype     float32 halves the live tensor; log_ndtr is checked against
            float64 before it is trusted.
  device    MPS if present. The inner object is a dense (chunk, Q, QZ, K)
            contraction, which is what the GPU is for.

T is scaled as 50 winners per class, the realistic validation-set regime
(a 50k set at K = 1000), which is the whole reason the loadings are tied to
the classifier's own class embeddings: parameters are DIM*(RANK+1)+1,
independent of K.
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import qmc
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

DIM, RANK = 32, 2
QZ = 9
N_FACTOR = 64
SEED = 11
MAXITER = 40


def factor_nodes(rank, n, seed=0):
    """Scrambled-Sobol nodes for N(0, I_rank), equal weights."""
    m = int(np.ceil(np.log2(n)))
    u = qmc.Sobol(rank, scramble=True, seed=seed).random_base2(m)
    return ndtri(np.clip(u, 1e-9, 1 - 1e-9))


def build(dev, dtype):
    F = torch.tensor(factor_nodes(RANK, N_FACTOR, SEED), device=dev, dtype=dtype)
    LWF = torch.full((len(F),), -np.log(len(F)), device=dev, dtype=dtype)
    nz, wz = hermegauss(QZ)
    wz = wz / wz.sum()
    return F, LWF, torch.tensor(nz, device=dev, dtype=dtype), \
        torch.log(torch.tensor(wz, device=dev, dtype=dtype))


def log_lik(mu, y, V, logD, nodes, chunk):
    F, LWF, NZ, LWZ = nodes
    D = torch.exp(logD)
    sd = torch.sqrt(D)
    total = 0.0
    for a in range(0, len(y), chunk):
        m, yy = mu[a:a + chunk], y[a:a + chunk]
        t = len(yy)
        mu_y = m.gather(1, yy[:, None]).squeeze(1)
        dv = V[yy][:, None, :] - V[None, :, :]              # (t, K, r)
        proj = torch.einsum("tkr,qr->tqk", dv, F)           # (t, Q, K)
        arg = ((mu_y[:, None] - m)[:, None, None, :]
               + proj[:, :, None, :]
               + (sd[yy][:, None] * NZ[None, :])[:, None, :, None]
               ) / sd[None, None, None, :]
        lc = torch.special.log_ndtr(arg)
        Q, QZn = lc.shape[1], lc.shape[2]
        lc = lc.scatter(3, yy[:, None, None, None].expand(t, Q, QZn, 1), 0.0)
        S = lc.sum(dim=3) + LWF[None, :, None] + LWZ[None, None, :]
        total = total + torch.logsumexp(S.reshape(t, -1), dim=1).sum()
    return total


def fit(mu, y, E, K, nodes, chunk, diag_only=False, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev, dt = mu.device, mu.dtype
    A = (0.01 * torch.randn(DIM, RANK, generator=g)).to(dev, dt).requires_grad_(not diag_only)
    b = torch.zeros(DIM, device=dev, dtype=dt, requires_grad=True)
    c = torch.zeros((), device=dev, dtype=dt, requires_grad=True)
    params = ([b, c] if diag_only else [A, b, c])
    opt = torch.optim.LBFGS(params, max_iter=MAXITER, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)

    def model():
        V = torch.zeros(K, RANK, device=dev, dtype=dt) if diag_only else E @ A
        return V - V.mean(dim=0, keepdim=True), E @ b + c

    def closure():
        opt.zero_grad()
        tot = 0.0
        for a in range(0, len(y), chunk):
            V, logD = model()
            nll = -log_lik(mu[a:a + chunk], y[a:a + chunk], V, logD, nodes, chunk)
            nll.backward()
            tot += float(nll.detach())
        return torch.tensor(tot)

    opt.step(closure)
    with torch.no_grad():
        V, logD = model()
        return V.detach().cpu().numpy().astype(np.float64), \
            np.exp(logD.detach().cpu().numpy().astype(np.float64))


def sample_races(mu, V, D, rng):
    f = rng.standard_normal((len(mu), V.shape[1]))
    u = mu + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal(mu.shape)
    return u.argmax(axis=1)


def contrast_err(Vh, Dh, Vt, Dt, K):
    M = np.eye(K) - np.ones((K, K)) / K
    A = M @ (Vh @ Vh.T + np.diag(Dh)) @ M
    B = M @ (Vt @ Vt.T + np.diag(Dt)) @ M
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def subspace_deg(Vh, Vt):
    qh, _ = np.linalg.qr(Vh); qt, _ = np.linalg.qr(Vt)
    s = np.clip(np.linalg.svd(qh.T @ qt, compute_uv=False), -1, 1)
    return float(np.degrees(np.arccos(s.min())))


def run_one(K, dev, dt, chunk, out_rows, tcap=None):
    rng = np.random.default_rng(SEED)
    E_np = rng.standard_normal((K, DIM)); E_np /= np.linalg.norm(E_np, axis=1, keepdims=True)
    A_true = rng.normal(0, 0.9, (DIM, RANK))
    V_true = E_np @ A_true; V_true -= V_true.mean(axis=0, keepdims=True)
    D_true = np.exp(E_np @ rng.normal(0, 0.5, DIM) - 0.3)
    T = 50 * K if tcap is None else min(50 * K, tcap)
    mu_tr = rng.normal(0, 2.0, (T, K)); y_tr = sample_races(mu_tr, V_true, D_true, rng)
    mu_h = rng.normal(0, 2.0, (10_000, K)); y_h = sample_races(mu_h, V_true, D_true, rng)
    nodes = build(dev, dt)
    E = torch.tensor(E_np, device=dev, dtype=dt)
    mu_t = torch.tensor(mu_tr, device=dev, dtype=dt); y_t = torch.tensor(y_tr, device=dev)
    mu_ht = torch.tensor(mu_h, device=dev, dtype=dt); y_ht = torch.tensor(y_h, device=dev)
    t0 = time.perf_counter()
    Vh, Dh = fit(mu_t, y_t, E, K, nodes, chunk)
    t_fit = time.perf_counter() - t0
    _, Dd = fit(mu_t, y_t, E, K, nodes, chunk, diag_only=True)
    with torch.no_grad():
        tt = lambda x: torch.tensor(x, device=dev, dtype=dt)
        ll_true = float(log_lik(mu_ht, y_ht, tt(V_true), torch.log(tt(D_true)), nodes, chunk))
        ll_fit = float(log_lik(mu_ht, y_ht, tt(Vh), torch.log(tt(Dh)), nodes, chunk))
        ll_dg = float(log_lik(mu_ht, y_ht, torch.zeros(K, RANK, device=dev, dtype=dt),
                              torch.log(tt(Dd)), nodes, chunk))
    row = {"K": K, "T": T, "dim": DIM, "rank": RANK, "fit_seconds": round(t_fit, 1),
           "subspace_deg": round(subspace_deg(Vh, V_true), 3),
           "med_abs_logD_err": round(float(np.median(np.abs(np.log(Dh / D_true)))), 4),
           "contrast_frob_rel": round(contrast_err(Vh, Dh, V_true, D_true, K), 4),
           "gap_per_race": round((ll_true - ll_fit) / 10_000, 6),
           "gain_per_race": round((ll_fit - ll_dg) / 10_000, 6)}
    out_rows.append(row)
    print(f"  K={K:5d} T={T:6d}  {t_fit:6.0f}s  subspace {row['subspace_deg']:6.2f} deg  "
          f"med|dlogD| {row['med_abs_logD_err']:.4f}  contrast {row['contrast_frob_rel']:.4f}  "
          f"gap {row['gap_per_race']:+.5f}  gain vs diag {row['gain_per_race']:+.5f}", flush=True)
    return row


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", type=int, nargs="+", default=[100, 200, 500, 1000])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--tcap", type=int, default=None)
    ap.add_argument("--out", default="results_embedding_tied_sweep.csv")
    a = ap.parse_args()
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else a.device
    dt = torch.float32 if a.dtype == "float32" else torch.float64
    print(f"device={dev} dtype={a.dtype} factor_nodes={N_FACTOR} QZ={QZ} chunk={a.chunk} tcap={a.tcap}", flush=True)
    rows = []
    for K in a.Ks:
        try:
            run_one(K, dev, dt, a.chunk, rows, tcap=a.tcap)
        except Exception as e:
            print(f"  K={K} FAILED: {type(e).__name__}: {e}", flush=True)
        if rows:
            import csv
            with open(HERE / a.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {HERE / a.out}", flush=True)
