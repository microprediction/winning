"""Joint calibration of abilities AND variances from multiple rank
markets.

The (mu, sigma) parameterization has a two-dimensional gauge: a common
shift of mu, and a common scale of (mu, sigma) jointly, since scaling
every performance preserves every ordering. So 2n - 2 parameters face
m(n - 1) constraints from m markets: TWO markets (win and top-20, say)
match dimensions exactly, and four are overdetermined with variance as
genuine signal -- a volatile contestant wins more often than his top-20
rate alone suggests.

No concavity theorem covers the joint problem (the potential argument
is for mu at fixed D), so uniqueness is probed empirically by
multistart. Gauss-Newton in (mu, log sigma) on log-quote residuals:
the mu block of the Jacobian is analytic (top_k_jacobian), the
log-sigma block by central differences.
"""
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from winning.factor.topk import top_k_probabilities, top_k_jacobian  # noqa


def residuals(mu, s, targets, points):
    r = []
    for k, t in targets.items():
        q = top_k_probabilities(mu, k, D=np.exp(2 * s), points=points)
        r.append(np.log(q) - np.log(t))
    return np.concatenate(r)


def gn_fit(targets, n, points=513, starts=1, seed=0, max_iter=80,
           tol=1e-12):
    """Levenberg-Marquardt in (mu, log sigma), gauges fixed by
    centering both. The raw Gauss-Newton step explodes along the
    near-null sigma directions of a symmetric start (measured |step| up
    to 6e8), so the normal equations carry an adaptive Marquardt
    damping instead of a clip that would mangle the direction. Returns
    the best of `starts` runs and all solutions for the uniqueness
    probe."""
    rng = np.random.default_rng(seed)
    ks = sorted(targets)
    k_warm = ks[-1]
    sols = []
    for trial in range(starts):
        # warm start: invert the deepest market at unit variances
        s = rng.normal(0, 0.2, n) if trial else np.zeros(n)
        s -= s.mean()
        if trial:
            mu = rng.normal(0, 0.5, n)
        else:
            t = np.log(targets[k_warm])
            mu = -(t - t.mean()) / 2
        mu -= mu.mean()
        lam = 1e-3
        r = residuals(mu, s, targets, points)
        sse = float(r @ r)
        for it in range(1, max_iter + 1):
            rows_mu, rows_s, rhs = [], [], []
            D = np.exp(2 * s)
            for k in ks:
                q = top_k_probabilities(mu, k, D=D, points=points)
                rhs.append(np.log(q) - np.log(targets[k]))
                rows_mu.append(top_k_jacobian(mu, k, D=D, points=points)
                               / q[:, None])
                Js = np.empty((n, n))
                h = 1e-4
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = h
                    qp = top_k_probabilities(mu, k, D=np.exp(2 * (s + e)),
                                             points=points)
                    qm = top_k_probabilities(mu, k, D=np.exp(2 * (s - e)),
                                             points=points)
                    Js[:, j] = (np.log(qp) - np.log(qm)) / (2 * h)
                rows_s.append(Js)
            r = np.concatenate(rhs)
            A = np.hstack([np.vstack(rows_mu), np.vstack(rows_s)])
            AtA = A.T @ A
            Atr = A.T @ r
            dscale = np.maximum(np.diag(AtA), 1e-12)
            improved = False
            for _ in range(12):
                try:
                    step = np.linalg.solve(AtA + lam * np.diag(dscale),
                                           Atr)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dmu = step[:n] - step[:n].mean()
                ds = step[n:] - step[n:].mean()
                mu_t = mu - dmu
                s_t = s - ds
                mu_t -= mu_t.mean()
                s_t -= s_t.mean()
                r_t = residuals(mu_t, s_t, targets, points)
                sse_t = float(r_t @ r_t)
                if sse_t < sse:
                    mu, s = mu_t, s_t
                    gain = sse - sse_t
                    sse = sse_t
                    lam = max(lam / 3, 1e-10)
                    improved = True
                    break
                lam *= 10
            if not improved or (gain < tol * max(sse, 1e-15)):
                break
        sols.append((mu, s, sse, it))
    best = min(sols, key=lambda z: z[2])
    return best, sols


def align(mu, s, mu_star, s_star):
    """Undo the shift and scale gauges against the truth."""
    c = s_star.mean() - s.mean()
    mu_a = np.exp(c) * (mu - mu.mean()) + mu_star.mean()
    s_a = s - s.mean() + s_star.mean()
    return mu_a, s_a


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 30
    mu_star = rng.normal(0, 0.8, n)
    mu_star -= mu_star.mean()
    sigma_star = np.exp(rng.normal(0, 0.3, n))     # heterogeneous spread
    s_star = np.log(sigma_star)
    D_star = sigma_star ** 2
    out = {}

    # --- two exact markets: win and top-20 ---------------------------
    truth = {k: top_k_probabilities(mu_star, k, D=D_star)
             for k in (1, 5, 10, 20)}
    t0 = time.time()
    (mu, s, sse, it), sols = gn_fit({1: truth[1], 20: truth[20]}, n,
                                    starts=3, seed=5)
    mu_a, s_a = align(mu, s, mu_star, s_star)
    print(f"win + top-20, exact quotes: SSE {sse:.2e} in {it} iters "
          f"({time.time()-t0:.0f} s); recovery after gauge alignment: "
          f"max |mu err| {np.abs(mu_a-mu_star).max():.2e}, "
          f"max |log sigma err| {np.abs(s_a-s_star).max():.2e}")
    spread = max(np.abs((z[0] - z[0].mean()) - (sols[0][0]
                 - sols[0][0].mean())).max() for z in sols[1:])
    print(f"  multistart (3 starts): max mu disagreement {spread:.2e}, "
          f"SSEs {[f'{z[2]:.1e}' for z in sols]}")
    # implied middle markets from the two-market fit
    imp = {k: float(np.abs(top_k_probabilities(mu, k, D=np.exp(2*s))
                           - truth[k]).max()) for k in (5, 10)}
    print(f"  implied top-5/top-10 vs truth: {imp}")
    out["two_exact"] = dict(sse=sse, iters=it,
                            mu_err=float(np.abs(mu_a-mu_star).max()),
                            s_err=float(np.abs(s_a-s_star).max()),
                            multistart_spread=float(spread), implied=imp)

    # --- four noisy markets, variances free vs fixed-wrong ------------
    noisy = {}
    for k in (1, 5, 10, 20):
        e = rng.normal(0, 0.05, n)
        qn = truth[k] * np.exp(e - e.mean())
        noisy[k] = qn * (k / qn.sum())
    t0 = time.time()
    (mu4, s4, sse4, it4), _ = gn_fit(noisy, n, starts=1)
    mu4a, s4a = align(mu4, s4, mu_star, s_star)
    print(f"four noisy markets, variances FREE: SSE {sse4:.4f} in {it4} "
          f"iters ({time.time()-t0:.0f} s); mu err "
          f"{np.abs(mu4a-mu_star).max():.3f}, log sigma err "
          f"{np.abs(s4a-s_star).max():.3f} (quote noise 0.05)")
    out["four_noisy_free"] = dict(sse=sse4, iters=it4,
                                  mu_err=float(np.abs(mu4a-mu_star).max()),
                                  s_err=float(np.abs(s4a-s_star).max()))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
