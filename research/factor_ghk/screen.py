"""Per-node contender screening: at factor node f, a runner whose conditional
mean sits more than c idiosyncratic sd above the lowest upper edge of the
field has survival ~1 across the whole lattice window (Phi(-c); 6e-16 at c=8)
and can be skipped at that node, exactly.  The pass becomes O(n r) to screen
plus O(k_f L) to price the k_f contenders, instead of O(n L).

    PYTHONPATH=. python3 research/factor_ghk/screen.py --n 100000 --m 13 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np                                                    # noqa: E402
from scipy.special import log_ndtr                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_a as H                                                  # noqa: E402
import nodes_b as NB                                                  # noqa: E402

LOG2PI = np.log(2.0 * np.pi)


def bulk_hi(Mk, sk, lo, hi, delta=1e-12):
    """x where the winner cdf 1 - prod_j S_j(x) reaches 1 - delta, i.e.
    sum_j log S_j(x) = log delta.  Bisection; monotone."""
    target = np.log(delta)
    a, b = lo, hi
    for _ in range(50):
        mid = 0.5 * (a + b)
        if log_ndtr(-(mid - Mk) / sk).sum() > target:
            a = mid
        else:
            b = mid
    return b


def screened_pass(mu, sd, V, F, W, c=8.0, points=257, chunk=256, bulk=True, tail=False):
    """Min-wins race probabilities, all runners, with per-node screening.
    Stage 1: drop runners whose mass lies entirely above the lowest upper
    edge min_k(M_k + c sd_k).  Stage 2 (bulk=True): per node, the winner
    distribution's upper 1e-12 quantile is the top of the lattice window
    (as the engine's _bulk_window, but per node) and runners with no mass
    below it are dropped too.  Returns (p normalised, mean contenders per
    node, seconds screening, seconds pricing)."""
    n = len(mu); p = np.zeros(n); ks = []; t_scr = 0.0; t_pr = 0.0
    logtail = np.full(n, -np.inf)                                     # wavefront tail for skipped runners, log space
    up = c * sd
    for a0 in range(0, len(F), chunk):
        t = time.perf_counter()
        M = mu[None, :] + F[a0:a0 + chunk] @ V.T                       # (chunk, n)
        hi = (M + up[None, :]).min(1)                                   # (chunk,) lowest upper edge
        keep_mask = (M - up[None, :]) <= hi[:, None]                    # runner has mass below hi
        t_scr += time.perf_counter() - t
        t = time.perf_counter()
        for a in range(M.shape[0]):
            idx = np.flatnonzero(keep_mask[a])
            Mk = M[a, idx]; sk = sd[idx]
            lo = (Mk - c * sk).min(); top = hi[a]
            if bulk:
                top = bulk_hi(Mk, sk, lo, top)
                sub = (Mk - c * sk) <= top
                idx, Mk, sk = idx[sub], Mk[sub], sk[sub]
            ks.append(len(idx))
            x = np.linspace(lo, top, points); dx = x[1] - x[0]
            z = (x[None, :] - Mk[:, None]) / sk[:, None]
            logS = log_ndtr(-z)
            logf = -0.5 * z ** 2 - 0.5 * LOG2PI - np.log(sk)[:, None]
            logSfield = logS.sum(0)
            rest = np.exp(np.clip(logSfield[None, :] - logS, -745.0, 0.0))
            p[idx] += W[a0 + a] * dx * (np.exp(logf) * rest).sum(1)
            if tail:
                # winner distribution given f from the contenders: cdf G = 1 - prod S on the window;
                # its mean and sd give the "wavefront" horse; a skipped runner j (S_j ~ 1 here)
                # wins iff X_j < winner, so p_j(f) ~ Phi((m_W - M_j) / sqrt(sd_j^2 + s_W^2)).
                Sf = np.exp(logSfield)
                g = -np.diff(Sf); xm = 0.5 * (x[1:] + x[:-1]); gsum = g.sum()
                if gsum > 0:
                    m_W = (g * xm).sum() / gsum; s_W = np.sqrt(max((g * (xm - m_W) ** 2).sum() / gsum, 1e-300))
                    skipped = np.ones(n, bool); skipped[idx] = False
                    lt = np.log(W[a0 + a]) + log_ndtr((m_W - M[a, skipped]) / np.sqrt(sd[skipped] ** 2 + s_W ** 2))
                    logtail[skipped] = np.logaddexp(logtail[skipped], lt)
        t_pr += time.perf_counter() - t
    if tail:
        p = p + np.exp(logtail)
    return p / p.sum(), float(np.mean(ks)), t_scr, t_pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--m", type=int, nargs="+", default=[9, 11])
    ap.add_argument("--c", type=float, default=8.0)
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--no-bulk", action="store_true")
    args = ap.parse_args()
    bulk = not args.no_bulk
    mu, V, D, sharp = H.make_field(args.n, 3, 0.05, 3); sd = np.sqrt(D)
    truth, d_scr, tkey = NB.largen_truth(args.n)
    order = np.argsort(-truth); targets = list(order[:8]) + [int(np.argmin(np.abs(truth - q))) for q in (1e-2, 3e-3, 1e-3, 3e-4)]
    print(f"\n=== screened lattice n={args.n} (sharpness {sharp:.0f}); truth {tkey} certified to {d_scr:.1e}; c={args.c}, points={args.points} ===", flush=True)
    # validation against the engine at 2^9 nodes
    F = NB.sobol_nodes(3, 9, 0); W = np.full(len(F), 1 / len(F))
    t = time.perf_counter(); pe = NB.price(mu, V, D, F); te = time.perf_counter() - t
    ps, k, ts, tp = screened_pass(mu, sd, V, F, W, args.c, args.points, bulk=bulk)
    print(f"  2^9 nodes: engine {te:.1f}s ({te/len(F)*1e3:.1f} ms/pass); screened {ts+tp:.1f}s ({(ts+tp)/len(F)*1e3:.2f} ms/pass: screen {ts/len(F)*1e3:.2f} + price {tp/len(F)*1e3:.2f}); mean contenders {k:.0f} of {args.n}; max |screened - engine| {np.abs(ps - pe).max():.1e}", flush=True)
    for m in args.m:
        F = NB.sobol_nodes(3, m, 0); W = np.full(len(F), 1 / len(F))
        ps, k, ts, tp = screened_pass(mu, sd, V, F, W, args.c, args.points, bulk=bulk)
        err = np.abs(ps - truth).max(); rel = max(abs(ps[t_] - truth[t_]) / truth[t_] for t_ in targets)
        print(f"  screened Sobol 2^{m}: all-n max abs {err:.1e}, targets max rel {rel:.1e}; {ts+tp:.1f}s ({(ts+tp)/len(F)*1e3:.2f} ms/pass), mean contenders {k:.0f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
