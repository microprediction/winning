"""Scale test: embedding-tied (V, D) at K = 200 with 50 winners per class.

The free-parameter gate (run_win_only_identification.py) showed clean
identification at K = 12 but needs a few hundred winners per class.
Real classifiers have K in the hundreds-to-thousands and ~50 validation
winners per class, so the loadings must be tied: V = E A and
log D = E b + c, with E the classifier's own (frozen, known) class
embeddings. Parameters drop from O(K) to O(d(r+1)): here 96 + 1
against 400 free. This script tests whether 50 winners per class
suffice under tying.

Truth: K = 200, embed dim d = 32, rank r = 2. E has unit-norm rows.
V* = E A*, log D* = E b* + c*. Menus mu_t ~ N(0, 4) iid; winners from
the true race. Fit (A, b, c) by exact-gradient ML (same conditioning
formula as the gate, float32 fields, chunked); baseline is the tied
diagonal-only model (A = 0).

Reported: contrast-covariance relative Frobenius error, subspace
principal angle between col(V_hat) and col(V*), median |dlog D|,
held-out log-likelihood gap to truth and gain over the diagonal fit.

Run: python research/recalibration/run_embedding_tied_scale.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
K, DIM, RANK = 200, 32, 2
QF, QZ = 15, 15
T_TRAIN = 10_000          # 50 winners per class on average
T_HELD = 10_000
CHUNK = 48   # rank-2 field is (t,441,21,K); keep the live graph small
SEED = 11

nodes_f, wf = hermegauss(QF)
nodes_z, wz = hermegauss(QZ)
wf, wz = wf / wf.sum(), wz / wz.sum()
LW = torch.log(torch.tensor(np.outer(wf, wz)))


def sample_races(mu, V, D, rng):
    f = rng.standard_normal((len(mu), V.shape[1]))
    eps = rng.standard_normal(mu.shape)
    u = mu + f @ V.T + np.sqrt(D)[None, :] * eps
    return u.argmax(axis=1)


def log_lik(mu, y, V, logD):
    """Sum of log p_y under rank-r loadings V (K, r), chunked over races.

    Integrates the r-dim factor by GH product only for r <= 2 via
    per-race reduction: conditional on the winner's row, only the
    projection of f onto (v_y - v_j) directions matters; we use the
    full product rule on r dims (QF^r nodes) -- fine at r = 2.
    """
    D = torch.exp(logD)
    sd = torch.sqrt(D)
    r = V.shape[1]
    if r == 1:
        F = torch.tensor(nodes_f).reshape(-1, 1)
        WF = torch.log(torch.tensor(wf))
    else:
        g1, g2 = np.meshgrid(nodes_f, nodes_f, indexing="ij")
        F = torch.tensor(np.stack([g1.ravel(), g2.ravel()], axis=1))
        WF = torch.log(torch.tensor(np.outer(wf, wf).ravel()))
    NZt = torch.tensor(nodes_z)
    LWZ = torch.log(torch.tensor(wz))
    total = 0.0
    for a in range(0, len(y), CHUNK):
        m = mu[a:a + CHUNK]
        yy = y[a:a + CHUNK]
        t = len(yy)
        mu_y = m.gather(1, yy[:, None]).squeeze(1)
        v_y = V[yy]                                   # (t, r)
        sd_y = sd[yy]
        base = mu_y[:, None] - m                      # (t, K)
        dv = v_y[:, None, :] - V[None, :, :]          # (t, K, r)
        proj = torch.einsum("tkr,qr->tqk", dv, F)     # (t, Q, K)
        arg = (base[:, None, None, :]
               + proj[:, :, None, :]
               + (sd_y[:, None] * NZt[None, :])[:, None, :, None]
               ) / sd[None, None, None, :]
        lc = torch.special.log_ndtr(arg)              # (t, Q, QZ, K)
        Q = lc.shape[1]
        lc = lc.scatter(3, yy[:, None, None, None].expand(t, Q, QZ, 1), 0.0)
        S = lc.sum(dim=3) + WF[None, :, None] + LWZ[None, None, :]
        total = total + torch.logsumexp(S.reshape(t, -1), dim=1).sum()
    return total


def fit(mu, y, E, rank, seed=0, diag_only=False, maxiter=60):
    g = torch.Generator().manual_seed(seed)
    A = (0.01 * torch.randn(DIM, rank, generator=g)).requires_grad_(not diag_only)
    b = torch.zeros(DIM, requires_grad=True)
    c = torch.zeros((), requires_grad=True)
    params = ([b, c] if diag_only else [A, b, c])
    opt = torch.optim.LBFGS(params, max_iter=maxiter, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def model():
        V = E @ A if not diag_only else torch.zeros(K, rank)
        V = V - V.mean(dim=0, keepdim=True)
        logD = E @ b + c
        return V, logD

    def closure():
        # backward per chunk so only one chunk's autograd graph is alive
        opt.zero_grad()
        total = 0.0
        for a in range(0, len(y), CHUNK):
            V, logD = model()
            nll_c = -log_lik(mu[a:a + CHUNK], y[a:a + CHUNK], V, logD)
            nll_c.backward()
            total += float(nll_c)
        return torch.tensor(total)

    opt.step(closure)
    with torch.no_grad():
        V, logD = model()
        return V.detach().numpy(), np.exp(logD.detach().numpy())


def contrast_err(V_hat, D_hat, V_true, D_true):
    M = np.eye(K) - np.ones((K, K)) / K
    S_h = M @ (V_hat @ V_hat.T + np.diag(D_hat)) @ M
    S_t = M @ (V_true @ V_true.T + np.diag(D_true)) @ M
    return np.linalg.norm(S_h - S_t) / np.linalg.norm(S_t)


def subspace_angle(V_hat, V_true):
    qh, _ = np.linalg.qr(V_hat)
    qt, _ = np.linalg.qr(V_true)
    s = np.linalg.svd(qh.T @ qt, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))


def main():
    rng = np.random.default_rng(SEED)
    E_np = rng.standard_normal((K, DIM))
    E_np /= np.linalg.norm(E_np, axis=1, keepdims=True)
    A_true = rng.normal(0.0, 0.9, (DIM, RANK))
    b_true = rng.normal(0.0, 0.5, DIM)
    V_true = E_np @ A_true
    V_true -= V_true.mean(axis=0, keepdims=True)
    D_true = np.exp(E_np @ b_true - 0.3)
    print(f"K={K}, d={DIM}, rank={RANK}; ||v|| median "
          f"{np.median(np.linalg.norm(V_true, axis=1)):.2f}, "
          f"D range [{D_true.min():.2f}, {D_true.max():.2f}]")

    mu_tr = rng.normal(0.0, 2.0, (T_TRAIN, K))
    y_tr = sample_races(mu_tr, V_true, D_true, rng)
    mu_h = rng.normal(0.0, 2.0, (T_HELD, K))
    y_h = sample_races(mu_h, V_true, D_true, rng)

    E = torch.tensor(E_np)
    mu_t, y_t = torch.tensor(mu_tr), torch.tensor(y_tr)
    mu_ht, y_ht = torch.tensor(mu_h), torch.tensor(y_h)

    with torch.no_grad():
        ll_true = log_lik(mu_ht, y_ht, torch.tensor(V_true),
                          torch.log(torch.tensor(D_true))).item()

    t0 = time.perf_counter()
    V_hat, D_hat = fit(mu_t, y_t, E, RANK)
    t_fit = time.perf_counter() - t0
    _, D_diag = fit(mu_t, y_t, E, RANK, diag_only=True)

    with torch.no_grad():
        ll_fit = log_lik(mu_ht, y_ht, torch.tensor(V_hat),
                         torch.log(torch.tensor(D_hat))).item()
        ll_dg = log_lik(mu_ht, y_ht, torch.zeros(K, RANK),
                        torch.log(torch.tensor(D_diag))).item()

    ce = contrast_err(V_hat, D_hat, V_true, D_true)
    ang = subspace_angle(V_hat, V_true)
    dD = float(np.median(np.abs(np.log(D_hat / D_true))))
    gap = (ll_true - ll_fit) / T_HELD
    gain = (ll_fit - ll_dg) / T_HELD
    print(f"\nfit {t_fit:.0f}s: subspace angle {ang:.1f} deg, "
          f"med|dlogD| {dD:.4f}, contrast err {ce:.4f}")
    print(f"held-out: gap vs truth {gap:.6f}/race, "
          f"gain vs tied-diagonal {gain:.6f}/race")
    (HERE / "results_embedding_tied.csv").write_text(
        "K,d,rank,T,subspace_deg,med_abs_logD_err,contrast_frob_rel,"
        "gap_per_race,gain_per_race\n"
        f"{K},{DIM},{RANK},{T_TRAIN},{ang:.4f},{dD:.6f},{ce:.6f},"
        f"{gap:.8f},{gain:.8f}\n")
    print(f"wrote {HERE / 'results_embedding_tied.csv'}")


if __name__ == "__main__":
    main()
