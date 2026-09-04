"""Correlated Gaussian bandit: exact PoM vs its variational surrogates.

Max-reward convention. Arm means theta ~ N(0, Sigma0) with
Sigma0 = V V' + diag(D0), rank 2; pulling arm a returns
theta_a + N(0, sigma_n^2). After per-arm counts c and sums s, the
posterior is N(m_t, Sigma_t) with

    Sigma_t^{-1} = Sigma0^{-1} + diag(c)/sigma_n^2,

and the same double-Woodbury as exp1_stopping keeps Sigma_t EXACTLY
factor-plus-diagonal: Sigma0^{-1} = D0^{-1} - U S0 U' (U = D0^{-1}V),
so the posterior precision is diag(a) - U S0 U' with
a = d0^{-1} + c/sigma_n^2, and inverting back gives
Sigma_t = diag(1/a) + W W'.

Policies, all acting on the same posterior law:
  ts      Thompson: sample theta-tilde, pull its argmax.
  exact   probability matching on the EXACT PoM vector (engine race,
          max by negation). VAPOR Lemma 8 says E[TS occupancy] = PoM,
          so ts and exact have identical expected behavior -- their
          regret curves coinciding is a MEASURED CHECK of Lemma 8,
          not a win.
  vbos    sample from the ToSFiT Eq. 2 policy (marginals only).
  flite   sample from the F-LITE independence PoM (marginals only).

Measured:
  * cumulative Bayes regret over T rounds (mean over replications);
  * the approximation gap TV(pi_policy, PoM_exact) at checkpoints
    along each policy's own trajectory (ts's is 0 by Lemma 8; vbos
    and flite are the exhibit);
  * terminal identification: P(posterior-mean argmax = true best)
    and expected shortfall theta_max - theta_rec at T.
Engine certified against Monte Carlo argmax counts at checkpoints.
"""
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scipy.stats import norm                       # noqa: E402
from winning.factor import race_probabilities      # noqa: E402

N = 50
RHO = 2
T = 200
REPS = 100
SIGMA_N = 1.0
GAP_EVERY = 10
RNG_GLOBAL = np.random.default_rng(20260902)


def make_prior(kind, c):
    u = np.zeros((N, RHO))
    if kind == "clusters":
        u[: N // 2, 0] = 1.0
        u[N // 2:, 1] = 1.0
    V = np.sqrt(c) * u
    d0 = np.maximum(1.0 - (V ** 2).sum(1), 1e-9)
    return V, d0


def posterior(V, d0, counts, sums):
    """N(m, diag(d) + W W') after per-arm counts/sums, prior mean 0."""
    Dinv = 1.0 / d0
    M0 = np.eye(RHO) + (V.T * Dinv) @ V
    S0 = np.linalg.inv(M0)                       # Sigma0^-1 = D^-1 - U S0 U'
    U = V * Dinv[:, None]
    a = Dinv + counts / SIGMA_N ** 2
    Ua = U / a[:, None]
    C = np.linalg.inv(np.linalg.inv(S0) - U.T @ Ua)
    evals, evecs = np.linalg.eigh(C)
    W = Ua @ (evecs * np.sqrt(np.maximum(evals, 0.0)))
    d = 1.0 / a
    h = sums / SIGMA_N ** 2                      # precision-weighted data
    m = h / a + Ua @ (C @ (Ua.T @ h))
    return m, W, d


def exact_pom(m, W, d):
    return race_probabilities(-m, V=-W, D=d)     # max race by negation


def flite_pom(m, s):
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if norm.cdf((m - mid) / s).sum() > 1 else \
            (lo, mid)
    q = norm.cdf((m - 0.5 * (lo + hi)) / s)
    return q / q.sum()


def vbos_policy(m, s):
    def v(cv):
        return np.exp(-(np.sqrt(cv ** 2 + 4.0) - cv) ** 2 / 8.0)
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if v((m - mid) / s).sum() > 1 else (lo, mid)
    p = v((m - 0.5 * (lo + hi)) / s)
    return p / p.sum()


def run_config(kind, c, rng):
    V, d0 = make_prior(kind, c)
    policies = ("ts", "exact", "vbos", "flite")
    regret = {p: np.zeros(T) for p in policies}
    tv = {p: {} for p in policies}
    ident = {p: [] for p in policies}
    shortfall = {p: [] for p in policies}
    cert_worst = 0.0
    for rep in range(REPS):
        z0 = rng.normal(size=RHO)
        theta = V @ z0 + rng.normal(0, np.sqrt(d0))
        best_val = theta.max()
        for pol in policies:
            counts = np.zeros(N)
            sums = np.zeros(N)
            for t in range(T):
                m, W, d = posterior(V, d0, counts, sums)
                sd = np.sqrt(d + (W ** 2).sum(1))
                need_gap = (t % GAP_EVERY == 0)
                if pol == "ts":
                    draw = (m + W @ rng.normal(size=W.shape[1])
                            + rng.normal(0, np.sqrt(d)))
                    a = int(draw.argmax())
                    pi = None
                elif pol == "exact":
                    p = exact_pom(m, W, d)
                    a = int(rng.choice(N, p=np.maximum(p, 0)
                                       / np.maximum(p, 0).sum()))
                    pi = p
                elif pol == "vbos":
                    pi = vbos_policy(m, sd)
                    a = int(rng.choice(N, p=pi))
                else:
                    pi = flite_pom(m, sd)
                    a = int(rng.choice(N, p=pi))
                if need_gap and pol in ("vbos", "flite"):
                    p_ex = exact_pom(m, W, d)
                    tv[pol].setdefault(t, []).append(
                        0.5 * np.abs(pi - p_ex).sum())
                    if rep == 0 and t == 50:
                        draws = (m[None, :]
                                 + rng.normal(size=(200_000, W.shape[1]))
                                 @ W.T
                                 + rng.normal(size=(200_000, N))
                                 * np.sqrt(d))
                        counts_mc = np.bincount(draws.argmax(1),
                                                minlength=N) / 200_000
                        cert_worst = max(cert_worst, 0.5 * np.abs(
                            counts_mc - p_ex).sum())
                regret[pol][t] += best_val - theta[a]
                counts[a] += 1
                sums[a] += theta[a] + rng.normal(0, SIGMA_N)
            m, W, d = posterior(V, d0, counts, sums)
            rec = int(m.argmax())
            ident[pol].append(int(rec == int(theta.argmax())))
            shortfall[pol].append(best_val - theta[rec])
    out = {}
    for p in policies:
        out[p] = dict(
            cum_regret=float(regret[p].sum() / REPS),
            ident=float(np.mean(ident[p])),
            shortfall=float(np.mean(shortfall[p])),
            tv_by_t={str(t): float(np.mean(v))
                     for t, v in sorted(tv[p].items())} if tv[p] else {},
        )
    out["mc_certificate_worst_tv"] = float(cert_worst)
    return out


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    for kind, c in (("independent", 0.0), ("clusters", 0.75)):
        rng = np.random.default_rng(hash((kind, 99)) % 2 ** 31)
        results[f"{kind}"] = run_config(kind, c, rng)
        r = results[f"{kind}"]
        print(f"[{kind} c={c}] cert TV {r['mc_certificate_worst_tv']:.4f}")
        for p in ("ts", "exact", "vbos", "flite"):
            row = r[p]
            tvs = row["tv_by_t"]
            tvtxt = (" tv@10/100/190 "
                     + "/".join(f"{tvs.get(str(t), float('nan')):.3f}"
                                for t in (10, 100, 190))) if tvs else ""
            print(f"  {p:6s} regret {row['cum_regret']:7.2f}  "
                  f"ident {row['ident']:.3f}  "
                  f"shortfall {row['shortfall']:.4f}{tvtxt}")
    print(f"total {time.time() - t0:.0f}s")
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
