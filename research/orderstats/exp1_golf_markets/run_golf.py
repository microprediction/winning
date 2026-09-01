"""Golf-market calibration: win / top-5 / top-10 / top-20 quotes, one
race model.

Single-market calibration is exact and theorem-backed (E[sum of k
smallest] is concave with gradient q^(k)). Four markets against n-1
abilities is overdetermined, so two experiments:

1. CONSISTENCY: calibrate on one market, price the others. With exact
   (model-generated) quotes every cross-market implication must hold to
   solver precision -- the race model's version of the Harville chain.
2. POOLED FIT: perturb all four markets with independent noise (the
   bookmaker's vig and the bettor's rounding), then maximize the
   lambda-weighted sum of potentials -- still concave, unique optimum --
   and report per-market fit against both the noisy quotes and the
   noiseless truth.
"""
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from winning.factor.topk import top_k_probabilities, top_k_jacobian  # noqa


def calibrate(targets, D, points=513, tol=1e-9, max_iter=300):
    """targets: dict k -> (weight, q_star). Damped diagonal iteration on
    the lambda-weighted residual, slopes from the summed Jacobian
    diagonals."""
    ks = sorted(targets)
    n = len(next(iter(targets.values()))[1])
    wsum = sum(w for w, _ in targets.values())
    mu = np.zeros(n)
    alpha, prev = 0.7, np.inf
    for it in range(1, max_iter + 1):
        q = {k: top_k_probabilities(mu, k, D=D, points=points) for k in ks}
        # the concave compromise's gradient: the lambda-weighted LINEAR
        # residual (a log residual has no zero when the targets are
        # inconsistent, and an iteration chasing one wanders forever)
        g = sum(w * (q[k] - t) for k, (w, t) in targets.items()) / wsum
        g -= g.mean()
        worst = float(np.abs(g).max())
        if worst < tol:
            return mu, it, worst
        if worst > prev:
            alpha = max(alpha / 2, 0.05)
        prev = worst
        d = sum(w * np.diag(top_k_jacobian(mu, k, D=D, points=points))
                for k, (w, t) in targets.items()) / wsum
        d = np.minimum(d, -1e-9)
        mu = mu - np.clip(alpha * g / d, -2, 2)
        mu -= mu.mean()
    return mu, max_iter, worst


def calibrate_log_ls(targets, D, points=513, tol=1e-10, max_iter=60):
    """Gauss-Newton least squares in LOG quotes: the right statistical
    objective for multiplicative quote noise. No global concavity
    theorem (unlike the potential compromise), but the exact Jacobians
    make the steps cheap and it converges in a handful of iterations on
    every field tried. Gauge fixed by centering."""
    ks = sorted(targets)
    n = len(next(iter(targets.values()))[1])
    mu = np.zeros(n)
    prev = np.inf
    alpha = 1.0
    for it in range(1, max_iter + 1):
        rows = []
        rhs = []
        sse = 0.0
        for k, (w, t) in targets.items():
            q = top_k_probabilities(mu, k, D=D, points=points)
            r = np.log(q) - np.log(t)
            sse += w * float(r @ r)
            Jl = np.sqrt(w) * top_k_jacobian(mu, k, D=D, points=points) \
                / q[:, None]
            rows.append(Jl)
            rhs.append(np.sqrt(w) * r)
        if abs(prev - sse) < tol * max(sse, 1e-12):
            return mu, it, sse
        prev = sse
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        # Gauss-Newton step on contrasts (center the step; the common
        # shift is the gauge)
        step, *_ = np.linalg.lstsq(A, b, rcond=None)
        step -= step.mean()
        mu = mu - alpha * np.clip(step, -2, 2)
        mu -= mu.mean()
    return mu, max_iter, sse


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 30                                   # a golf field after the cut
    ks = [1, 5, 10, 20]
    mu_star = rng.normal(0, 0.8, n)
    mu_star -= mu_star.mean()
    D = 0.6 + 0.8 * rng.random(n)            # uneven consistency
    truth = {k: top_k_probabilities(mu_star, k, D=D) for k in ks}
    out = {}

    # 1: calibrate on top-20 alone, price the rest
    t0 = time.time()
    mu20, it20, res20 = calibrate({20: (1.0, truth[20])}, D)
    implied = {k: top_k_probabilities(mu20, k, D=D) for k in (1, 5, 10)}
    cons = {k: float(np.abs(implied[k] - truth[k]).max()) for k in implied}
    print(f"calibrate on top-20 alone: {it20} iters, residual {res20:.1e},"
          f" {time.time()-t0:.1f} s")
    print("  implied win/top-5/top-10 vs truth:",
          {k: f"{v:.2e}" for k, v in cons.items()})
    out["consistency"] = dict(iters=it20, residual=res20, implied=cons)

    # 2: pooled fit to four NOISY markets (multiplicative log noise)
    noisy = {}
    for k in ks:
        e = rng.normal(0, 0.05, n)           # 5 percent quote noise
        qn = truth[k] * np.exp(e - e.mean())
        qn = qn * (k / qn.sum())
        noisy[k] = qn
    targets = {k: (1.0, noisy[k]) for k in ks}
    t0 = time.time()
    mu_fit, itp, resp = calibrate(targets, D, tol=1e-9)
    fit = {k: top_k_probabilities(mu_fit, k, D=D) for k in ks}
    to_noisy = {k: float(np.abs(np.log(fit[k]) - np.log(noisy[k])).max())
                for k in ks}
    to_truth = {k: float(np.abs(np.log(fit[k]) - np.log(truth[k])).max())
                for k in ks}
    print(f"pooled fit to 4 noisy markets: {itp} iters, gradient"
          f" {resp:.1e}, {time.time()-t0:.1f} s")
    print("  max |log fit - log quote| per market:",
          {k: f"{v:.3f}" for k, v in to_noisy.items()})
    print("  max |log fit - log truth| per market:",
          {k: f"{v:.3f}" for k, v in to_truth.items()})
    print("  mu recovery vs truth: max err "
          f"{np.abs(mu_fit - mu_star).max():.3f}"
          f" (quote noise was 0.05 in logs)")
    out["pooled"] = dict(iters=itp, residual=resp, to_noisy=to_noisy,
                         to_truth=to_truth,
                         mu_err=float(np.abs(mu_fit - mu_star).max()))

    # 3: log least squares across the same four noisy markets
    t0 = time.time()
    mu_ls, itl, sse = calibrate_log_ls(targets, D)
    fit = {k: top_k_probabilities(mu_ls, k, D=D) for k in ks}
    ln = {k: float(np.abs(np.log(fit[k]) - np.log(noisy[k])).max())
          for k in ks}
    lt = {k: float(np.abs(np.log(fit[k]) - np.log(truth[k])).max())
          for k in ks}
    print(f"log least squares, 4 noisy markets: {itl} iters, SSE {sse:.4f},"
          f" {time.time()-t0:.1f} s")
    print("  max |log fit - log quote| per market:",
          {k: f"{v:.3f}" for k, v in ln.items()})
    print("  max |log fit - log truth| per market:",
          {k: f"{v:.3f}" for k, v in lt.items()})
    print("  mu recovery vs truth: max err "
          f"{np.abs(mu_ls - mu_star).max():.3f}")
    out["log_ls"] = dict(iters=itl, sse=sse, to_noisy=ln, to_truth=lt,
                         mu_err=float(np.abs(mu_ls - mu_star).max()))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
