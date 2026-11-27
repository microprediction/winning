"""Candidate cure for slow convergence of the factor inversion under
strong loadings: correct the loading directions separately each sweep.

The diagonal iteration uses each contestant's own slope only. A shift of
the abilities along a loading column moves the probabilities little when
the loadings are strong, so the error along those columns shrinks slowly
(run_convergence_battery.py: median sweeps rise from about 10 at loading
scale 0.25 to over 50 at scale 2).

Rules compared, all on the loop of abilities_from_race with the sign
rule of run_sign_rule_barrage.py as the step multiplier:

  sign       the diagonal step alone
  galerkin   before the diagonal step, a step U c along the centred
             loading columns U (n by rank). G = d log p / d mu . U is
             taken by one forward difference per column, and c solves
             U' diag(p) G c = -U' diag(p) resid. The diagonal step then
             acts on the predicted residual resid + G c.
  lstsq      the same with c the least-squares solution of G c = -resid
  anderson   Anderson acceleration (memory 5) of the diagonal step with
             multiplier 1, no loading directions

A sweep of galerkin or lstsq costs 1 + rank forward passes; the others
cost 1. The table reports sweeps and forward passes.

Fields and seeds are those of run_sign_rule_barrage.py, stage "large"
(so the sign column repeats that log) and stage "small" for n <= 20.

    nice -n 15 python run_factor_direction_correction.py [n ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np

SIZES = [int(a) for a in sys.argv[1:]] or [100, 1000]
_argv = sys.argv
sys.argv = ["x", "large", "3", "1"]
import run_sign_rule_barrage as B          # noqa: E402
sys.argv = _argv
from winning.factor.races import race_probabilities   # noqa: E402

TOL, CAP, EPS = 1e-8, 300, 1e-5
RULES = ("sign", "galerkin", "lstsq", "anderson")


def solve(target, kw, rule, cap=CAP):
    """Returns (sweeps, forward passes); sweeps is cap + 1 if stuck."""
    n = len(target)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    U = None
    if rule in ("galerkin", "lstsq") and "V" in kw:
        U = np.array(kw["V"], dtype=float).reshape(n, -1)
        U = U - U.mean(axis=0)
        U = U / np.maximum(np.abs(U).max(axis=0), 1e-300)
    alpha, prev, passes = np.ones(n), None, 0
    hist_x, hist_f = [], []
    for it in range(cap):
        phat, sl = race_probabilities(mu, return_slopes=True, **kw)
        passes += 1
        logp = np.log(np.maximum(phat, 1e-300))
        resid = logp - logt
        if float(np.abs(resid).max()) < TOL:
            return it + 1, passes
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        if rule == "anderson":
            lim = np.minimum(2.0, 10.0 * np.abs(resid))
            f = -np.clip(resid / dlogp, -lim, lim)       # g(mu) - mu
            hist_x.append(mu.copy())
            hist_f.append(f.copy())
            hist_x, hist_f = hist_x[-6:], hist_f[-6:]
            new = mu + f
            if len(hist_f) > 1:
                dF = np.stack([hist_f[i + 1] - hist_f[i]
                               for i in range(len(hist_f) - 1)], axis=1)
                dX = np.stack([hist_x[i + 1] - hist_x[i]
                               for i in range(len(hist_x) - 1)], axis=1)
                gam = np.linalg.lstsq(dF, f, rcond=1e-10)[0]
                new = mu + f - (dX + dF) @ gam
            mu = new - new.mean()
            continue
        if prev is not None:
            flip = resid * prev < 0
            alpha = np.where(flip, np.maximum(alpha * 0.5, 0.05),
                             np.minimum(alpha * 1.5, 1.0))
        prev = resid
        step = np.zeros(n)
        if U is not None:
            G = np.empty_like(U)
            for c in range(U.shape[1]):
                pc = race_probabilities(mu + EPS * U[:, c], **kw)
                passes += 1
                G[:, c] = (np.log(np.maximum(pc, 1e-300)) - logp) / EPS
            if rule == "galerkin":
                A = U.T @ (phat[:, None] * G)
                b = -U.T @ (phat * resid)
                coef = np.linalg.lstsq(A, b, rcond=1e-12)[0]
            else:
                coef = np.linalg.lstsq(G, -resid, rcond=1e-12)[0]
            step = U @ coef
            big = np.abs(step).max()
            if big > 2.0:                       # same bound as the clip
                step, coef = step * (2.0 / big), coef * (2.0 / big)
            resid = resid + G @ coef            # predicted after U c
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu + step - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return cap + 1, passes


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print(f"cells: failures at 60 sweeps / median sweeps / largest sweeps / "
          f"median forward passes; {CAP} sweeps allowed")
    print(f"{'class':14s} {'n':>6s} {'scale':>5s} {'runs':>4s}  "
          + "  ".join(f"{r:>17s}" for r in RULES))
    for n in SIZES:
        small = n <= 20
        B.S = 0 if small else 2
        B.SPREADS = (0.5, 1.0, 2.0, 3.0) if small else (1.0,)
        reps = 5 if small else 3
        for g in (1, 2):
            for j, scale in enumerate(B.SCALES):
                res = {r: [] for r in RULES}
                for k in range(len(B.SPREADS)):
                    for r in range(reps):
                        mu, free, kw = B.field(g, n, j, k, r)
                        p = B.price(mu, kw)
                        for t in (p, free["dirichlet 1"]):
                            t = t / t.sum()
                            for rule in RULES:
                                res[rule].append(solve(t, kw, rule))
                cells = []
                for rule in RULES:
                    sw = np.array([x[0] for x in res[rule]])
                    ps = np.array([x[1] for x in res[rule]])
                    cells.append(f"{int(np.sum(sw > 60)):2d}/{int(np.median(sw)):3d}/"
                                 f"{sw.max():3d}/{int(np.median(ps)):4d}")
                print(f"{B.CLASSES[g]:14s} {n:6d} {scale:5g} {len(res['sign']):4d}  "
                      + "  ".join(f"{c:>17s}" for c in cells), flush=True)
