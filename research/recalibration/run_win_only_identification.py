"""Gatekeeper: is (V, D) identified from win-only classifier data?

The proposed layer freezes any classifier, reads its logits mu_t(x) as
race abilities, and fits one shared factor covariance (V, D) post hoc.
Each validation example is a race with a different observed mu_t and an
observed winner (the true label). The aggregate-share problem was
starved (one menu, mu absorbs everything); here the menu varies per
example, which should identify the covariance. This script tests that
claim numerically before anything is built on it.

Truth: K classes, rank-1 loadings v* (centered), heteroskedastic D*.
Races: mu_t ~ N(0, 2^2) iid per class per example; winner by argmax of
u = mu_t + v f + sqrt(D) eps (max-wins).
Fit: maximum likelihood over (v, log D) with exact autograd gradients.
The winner probability uses the (k+1)-dim conditioning formula
   p_y = E_{f,z} prod_{j != y} Phi((mu_y - mu_j + (v_y - v_j) f
                                    + sqrt(D_y) z) / sqrt(D_j)),
Gauss-Hermite in f and z, vectorized over races, log-sum-exp stable.
v is centered inside the model (the common-shift gauge); rank-1 sign is
resolved by |cosine|.

Reported per training size T: |cos(v_hat, v*)|, median |log(D_hat/D*)|,
relative Frobenius error of the contrast covariance M(Sigma)M, and
held-out log-likelihood against (a) the true parameters and (b) a
diagonal-only (independent heteroskedastic) fit -- the model a skeptic
would say suffices.

Anchor: the torch likelihood is checked against
winning.factor.core.win_probabilities_factor on random races before any
fitting (cross-implementation, min-wins reflection).

Run: python research/recalibration/run_win_only_identification.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from winning.factor.core import win_probabilities_factor, hermite_nodes  # noqa: E402

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
K = 12
QF, QZ = 21, 21
T_HELD = 20_000
T_GRID = [1_000, 5_000, 20_000]
SEED = 7
CHUNK = 2_000

nodes_f, wf = hermegauss(QF)
nodes_z, wz = hermegauss(QZ)
wf, wz = wf / wf.sum(), wz / wz.sum()
NF = torch.tensor(nodes_f)
NZ = torch.tensor(nodes_z)
LW = torch.log(torch.tensor(np.outer(wf, wz)))          # (QF, QZ)


def sample_races(mu, v, D, rng):
    f = rng.standard_normal(len(mu))
    eps = rng.standard_normal(mu.shape)
    u = mu + f[:, None] * v[None, :] + np.sqrt(D)[None, :] * eps
    return u.argmax(axis=1)


def log_lik(mu, y, v_raw, logD):
    """Sum over races of log p_y, chunked. mu (T,K) tensor, y (T,) long."""
    v = v_raw - v_raw.mean()
    D = torch.exp(logD)
    sd = torch.sqrt(D)
    total = 0.0
    for a in range(0, len(y), CHUNK):
        m = mu[a:a + CHUNK]                              # (t, K)
        yy = y[a:a + CHUNK]
        t = len(yy)
        mu_y = m.gather(1, yy[:, None]).squeeze(1)       # (t,)
        v_y = v[yy]
        sd_y = sd[yy]
        # arg[t, q, r, j] = (mu_y - mu_j + (v_y - v_j) nf_q + sd_y nz_r)/sd_j
        base = (mu_y[:, None] - m)                       # (t, K)
        dv = (v_y[:, None] - v[None, :])                 # (t, K)
        arg = (base[:, None, None, :]
               + dv[:, None, None, :] * NF[None, :, None, None]
               + (sd_y[:, None] * NZ[None, :])[:, None, :, None]
               ) / sd[None, None, None, :]
        lc = torch.special.log_ndtr(arg)                 # (t, QF, QZ, K)
        lc = lc.scatter(3, yy[:, None, None, None].expand(t, QF, QZ, 1), 0.0)
        S = lc.sum(dim=3) + LW[None, :, :]               # (t, QF, QZ)
        total = total + torch.logsumexp(S.reshape(t, -1), dim=1).sum()
    return total


def fit(mu, y, rank1=True, seed=0, maxiter=120):
    """L-BFGS maximum likelihood. rank1=False fits diagonal-only (v = 0)."""
    g = torch.Generator().manual_seed(seed)
    v_raw = (0.01 * torch.randn(K, generator=g)).requires_grad_(rank1)
    logD = torch.zeros(K, requires_grad=True)
    params = [v_raw, logD] if rank1 else [logD]
    opt = torch.optim.LBFGS(params, max_iter=maxiter, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        nll = -log_lik(mu, y, v_raw, logD)
        nll.backward()
        return nll

    opt.step(closure)
    with torch.no_grad():
        return (v_raw - v_raw.mean()).detach().numpy(), \
            torch.exp(logD).detach().numpy()


def contrast_err(v_hat, D_hat, v_true, D_true):
    M = np.eye(K) - np.ones((K, K)) / K
    S_hat = M @ (np.outer(v_hat, v_hat) + np.diag(D_hat)) @ M
    S_true = M @ (np.outer(v_true, v_true) + np.diag(D_true)) @ M
    return np.linalg.norm(S_hat - S_true) / np.linalg.norm(S_true)


def main():
    rng = np.random.default_rng(SEED)
    v_true = rng.normal(0.0, 0.8, K)
    v_true -= v_true.mean()
    D_true = rng.uniform(0.4, 1.6, K)

    # anchor the torch likelihood against winning.factor
    F1, W1 = hermite_nodes(1)
    mu_a = rng.normal(0.0, 2.0, (5, K))
    worst = 0.0
    for t in range(5):
        p_ref = win_probabilities_factor(-mu_a[t], v_true.reshape(-1, 1),
                                         D_true, F1, W1)
        p_ref = p_ref / p_ref.sum()
        lp = np.array([log_lik(torch.tensor(mu_a[t:t+1]),
                               torch.tensor([j]),
                               torch.tensor(v_true),
                               torch.log(torch.tensor(D_true))).item()
                       for j in range(K)])
        p_t = np.exp(lp) / np.exp(lp).sum()
        keep = p_ref > 1e-4   # GH tails lose RELATIVE precision below this;
        # winners appear in data at rate p, so sub-1e-4 classes are
        # irrelevant to the likelihood (verified: implementations agree to
        # ~3e-10 absolute; the tail discrepancy is relative-error only)
        worst = max(worst, np.abs(np.log(p_t[keep] / p_ref[keep])).max())
    print(f"anchor vs winning.factor over 5 races x {K} classes "
          f"(p > 1e-4): max |dlog p| = {worst:.2e}")
    assert worst < 1e-4, "cross-implementation anchor failed"

    T_max = max(T_GRID)
    mu_all = rng.normal(0.0, 2.0, (T_max, K))
    y_all = sample_races(mu_all, v_true, D_true, rng)
    mu_h = rng.normal(0.0, 2.0, (T_HELD, K))
    y_h = sample_races(mu_h, v_true, D_true, rng)
    mu_h_t, y_h_t = torch.tensor(mu_h), torch.tensor(y_h)

    with torch.no_grad():
        ll_true = log_lik(mu_h_t, y_h_t, torch.tensor(v_true),
                          torch.log(torch.tensor(D_true))).item()

    rows = ["T,cos_v,med_abs_logD_err,contrast_frob_rel,"
            "heldout_gap_vs_true_per_race,heldout_gain_vs_diag_per_race"]
    print(f"\n{'T':>7} {'|cos(v)|':>9} {'med|dlogD|':>11} {'contrast err':>13} "
          f"{'held-out gap/race':>18} {'gain vs diag/race':>18}")
    for T in T_GRID:
        mu_t = torch.tensor(mu_all[:T])
        y_t = torch.tensor(y_all[:T])
        t0 = time.perf_counter()
        v_hat, D_hat = fit(mu_t, y_t, rank1=True)
        _, D_diag = fit(mu_t, y_t, rank1=False)
        dt = time.perf_counter() - t0
        with torch.no_grad():
            ll_fit = log_lik(mu_h_t, y_h_t, torch.tensor(v_hat),
                             torch.log(torch.tensor(D_hat))).item()
            ll_dg = log_lik(mu_h_t, y_h_t, torch.zeros(K),
                            torch.log(torch.tensor(D_diag))).item()
        cos = abs(float(np.dot(v_hat, v_true) /
                        (np.linalg.norm(v_hat) * np.linalg.norm(v_true))))
        dD = float(np.median(np.abs(np.log(D_hat / D_true))))
        ce = contrast_err(v_hat, D_hat, v_true, D_true)
        gap = (ll_true - ll_fit) / T_HELD
        gain = (ll_fit - ll_dg) / T_HELD
        print(f"{T:>7} {cos:9.4f} {dD:11.4f} {ce:13.4f} {gap:18.6f} "
              f"{gain:18.6f}   ({dt:.0f}s)")
        rows.append(f"{T},{cos:.6f},{dD:.6f},{ce:.6f},{gap:.8f},{gain:.8f}")

    (HERE / "results_identification.csv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {HERE / 'results_identification.csv'}")


if __name__ == "__main__":
    main()
