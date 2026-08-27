"""First real-data Thurstonian recalibration: CIFAR-100, frozen resnet56.

Data: data/cifar100_logits.npz (10k test logits from a public pretrained
cifar100_resnet56, measured top-1 0.7262). Split 5k calibration / 5k
evaluation, stratified. The classifier's own head weights are the class
embeddings E (d = 64).

Models compared on the evaluation half:
  softmax     raw logits
  temp        temperature scaling, tau fit on calibration NLL
  race-diag   mu = logits/s, tied log D = E b + c, V = 0
  race-r2     mu = logits/s, V = E A (rank 2), tied log D
Metrics: NLL per example, 15-bin top-label ECE, Brier (one-hot), and
the two structure probes:
  superclass  correlation between fitted VV' off-diagonal entries and
              the same-superclass indicator (does V rediscover the 20
              CIFAR superclasses it was never shown?)
  restricted  per-example menu restricted to the true superclass's 5
              fine labels; conditional NLL under renormalized
              temperature-softmax vs exact race deletion (IIA vs not).

Fitting uses the (k+1)-dim conditioning likelihood with exact autograd
gradients (winner = true label). Full evaluation vectors and restricted
menus use winning.factor.core.win_probabilities_factor (min-wins at
-mu, keep= for deletions).

Run: python research/recalibration/run_cifar_recalibration.py
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
RANK = 2
QF, QZ = 15, 15
CHUNK = 96   # K=100 rank-2 field; keep the live graph small
SEED = 3

nf, wf = hermegauss(QF)
nz, wz = hermegauss(QZ)
wf, wz = wf / wf.sum(), wz / wz.sum()


def make_loglik(K):
    g1, g2 = np.meshgrid(nf, nf, indexing="ij")
    F = torch.tensor(np.stack([g1.ravel(), g2.ravel()], axis=1))
    WF = torch.log(torch.tensor(np.outer(wf, wf).ravel()))
    NZt = torch.tensor(nz)
    LWZ = torch.log(torch.tensor(wz))

    def log_lik(mu, y, V, logD):
        D = torch.exp(logD)
        sd = torch.sqrt(D)
        total = 0.0
        for a in range(0, len(y), CHUNK):
            m = mu[a:a + CHUNK]
            yy = y[a:a + CHUNK]
            t = len(yy)
            mu_y = m.gather(1, yy[:, None]).squeeze(1)
            v_y = V[yy]
            sd_y = sd[yy]
            base = mu_y[:, None] - m
            dv = v_y[:, None, :] - V[None, :, :]
            proj = torch.einsum("tkr,qr->tqk", dv, F)
            arg = (base[:, None, None, :]
                   + proj[:, :, None, :]
                   + (sd_y[:, None] * NZt[None, :])[:, None, :, None]
                   ) / sd[None, None, None, :]
            lc = torch.special.log_ndtr(arg)
            Q = lc.shape[1]
            lc = lc.scatter(3, yy[:, None, None, None].expand(t, Q, QZ, 1), 0.0)
            S = lc.sum(dim=3) + WF[None, :, None] + LWZ[None, None, :]
            total = total + torch.logsumexp(S.reshape(t, -1), dim=1).sum()
        return total

    return log_lik


def fit_race(logits, y, E, rank, diag_only=False, maxiter=100):
    K, d = E.shape
    log_lik = make_loglik(K)
    mu_raw = torch.tensor(logits)
    y_t = torch.tensor(y)
    E_t = torch.tensor(E)
    A = (0.01 * torch.randn(d, rank,
                            generator=torch.Generator().manual_seed(0))
         ).requires_grad_(not diag_only)
    b = torch.zeros(d, requires_grad=True)
    c = torch.zeros((), requires_grad=True)
    log_s = torch.zeros((), requires_grad=True)
    params = [b, c, log_s] if diag_only else [A, b, c, log_s]
    opt = torch.optim.LBFGS(params, max_iter=maxiter, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-8, tolerance_change=1e-11)

    def model():
        V = E_t @ A if not diag_only else torch.zeros(K, rank)
        V = V - V.mean(dim=0, keepdim=True)
        return V, E_t @ b + c

    def closure():
        # backward per chunk so only one chunk's autograd graph is alive
        opt.zero_grad()
        total = 0.0
        for a in range(0, len(y_t), CHUNK):
            V, logD = model()
            nll_c = -log_lik(mu_raw[a:a + CHUNK] / torch.exp(log_s),
                             y_t[a:a + CHUNK], V, logD)
            nll_c.backward()
            total += float(nll_c)
        return torch.tensor(total)

    opt.step(closure)
    with torch.no_grad():
        V, logD = model()
        return (V.numpy(), np.exp(logD.numpy()), float(torch.exp(log_s)))


def race_vectors(logits, V, D, s, F, W, keep=None):
    """Full win-probability vectors per example via the shared field."""
    out = np.zeros((len(logits), V.shape[0] if keep is None else len(keep)))
    for i, lg in enumerate(logits):
        p = win_probabilities_factor(-lg / s, V, D, F, W, keep=keep)
        out[i] = p / p.sum()
    return out


def metrics(P, y):
    n = len(y)
    py = np.maximum(P[np.arange(n), y], 1e-300)
    nll = float(-np.mean(np.log(py)))
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.clip((conf * 15).astype(int), 0, 14)
    ece = 0.0
    for b_ in range(15):
        m = bins == b_
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    onehot = np.zeros_like(P)
    onehot[np.arange(n), y] = 1.0
    brier = float(np.mean(((P - onehot) ** 2).sum(axis=1)))
    return nll, float(ece), brier


def main():
    dat = np.load(HERE / "data" / "cifar100_logits.npz")
    logits, y = dat["logits"].astype(np.float64), dat["labels"]
    E = dat["head_weight"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    coarse = dat["coarse_of_fine"]
    K = logits.shape[1]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    cal, ev = idx[:5000], idx[5000:]

    # temperature baseline
    lg_c = torch.tensor(logits[cal]); y_c = torch.tensor(y[cal])
    logt = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([logt], max_iter=60, line_search_fn="strong_wolfe")

    def tclos():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            lg_c / torch.exp(logt), y_c, reduction="sum")
        loss.backward()
        return loss
    opt.step(tclos)
    tau = float(torch.exp(logt))

    t0 = time.perf_counter()
    V2, D2, s2 = fit_race(logits[cal], y[cal], E, RANK)
    Vd, Dd, sd_ = fit_race(logits[cal], y[cal], E, RANK, diag_only=True)
    print(f"fits done in {time.perf_counter()-t0:.0f}s; tau={tau:.3f}, "
          f"s(race-r2)={s2:.3f}, s(diag)={sd_:.3f}")

    F2, W2 = hermite_nodes(RANK)
    def softmax(lg, t=1.0):
        z = lg / t; z -= z.max(axis=1, keepdims=True)
        e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

    lg_e, y_e = logits[ev], y[ev]
    rows = ["model,nll,ece,brier"]
    P_temp = softmax(lg_e, tau)
    for name, P in [("softmax", softmax(lg_e)),
                    ("temp", P_temp),
                    ("race-diag", race_vectors(lg_e, np.zeros((K, RANK)), Dd,
                                               sd_, F2, W2)),
                    ("race-r2", race_vectors(lg_e, V2, D2, s2, F2, W2))]:
        nll, ece, brier = metrics(P, y_e)
        print(f"{name:10s} NLL {nll:.4f}  ECE {ece:.4f}  Brier {brier:.4f}")
        rows.append(f"{name},{nll:.6f},{ece:.6f},{brier:.6f}")

    # superclass probe: VV' off-diagonals vs same-superclass indicator
    G = V2 @ V2.T
    iu = np.triu_indices(K, 1)
    same = (coarse[:, None] == coarse[None, :])[iu].astype(float)
    r = np.corrcoef(G[iu], same)[0, 1]
    top_frac = same[np.argsort(G[iu])[-len(iu[0]) // 20:]].mean()
    print(f"superclass probe: corr(VV', same-super) = {r:.3f}; "
          f"top-5% VV' pairs same-super {top_frac:.1%} (base "
          f"{same.mean():.1%})")
    rows.append(f"superclass_corr,{r:.6f},,")
    rows.append(f"superclass_top5pct,{top_frac:.6f},{same.mean():.6f},")

    # restricted-menu probe: condition on the true superclass's 5 labels
    nll_iia, nll_race, n_used = 0.0, 0.0, 0
    for i in ev:
        keep = np.where(coarse == coarse[y[i]])[0]
        pos = int(np.where(keep == y[i])[0][0])
        p_iia = softmax(logits[i][keep][None, :] / tau)[0]
        p_rc = win_probabilities_factor(-logits[i] / s2, V2, D2, F2, W2,
                                        keep=keep)
        p_rc = p_rc / p_rc.sum()
        nll_iia -= np.log(max(p_iia[pos], 1e-300))
        nll_race -= np.log(max(p_rc[pos], 1e-300))
        n_used += 1
    print(f"restricted-menu NLL/example: renormalized temp-softmax "
          f"{nll_iia/n_used:.4f} vs race deletion {nll_race/n_used:.4f}")
    rows.append(f"restricted_iia,{nll_iia/n_used:.6f},,")
    rows.append(f"restricted_race,{nll_race/n_used:.6f},,")

    (HERE / "results_cifar.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {HERE / 'results_cifar.csv'}")


if __name__ == "__main__":
    main()
