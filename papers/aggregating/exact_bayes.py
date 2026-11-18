"""Exact-Bayes diagnostic: particles from the prior, exact race likelihoods by
vectorised Gauss-Hermite. Compares, per race, the log-loss of
  bayes   E_post p(x)                (the Aggregating Algorithm at eta=1)
  plugB   p(E_post x)                (plug-in at the exact posterior mean)
  lapmix  p_{I+S}(m) Laplace (one Newton step, full block)
  lapplug p(m)       Laplace
"""
import sys
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr


from winning.factor.races import race_probabilities

z, wz = hermegauss(80); wz = wz / wz.sum()


def race_iid(X):
    """X (..., k) abilities, unit-variance independent noise, min wins."""
    # P(i wins) = E_z[ prod_{j!=i} (1 - Phi(z + x_i - x_j)) ], Y_i = x_i + z
    Xi = X[..., :, None, None]; Xj = X[..., None, :, None]
    S = 1.0 - ndtr(z + Xi - Xj)                      # (..., k, k, Q)
    k = X.shape[-1]
    S[..., np.arange(k), np.arange(k), :] = 1.0
    return (np.prod(S, axis=-2) * wz).sum(-1)


def run(seed, n=6, k=3, T=40, M=60000):
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal(n)
    P = rng.standard_normal((M, n)); logw = np.zeros(M)
    m = np.zeros(n); S = np.eye(n)
    L = dict(bayes=0.0, plugB=0.0, lapmix=0.0, lapplug=0.0)
    for _ in range(T):
        F = rng.choice(n, k, replace=False)
        p = race_iid(truth[F]); y = int(rng.choice(k, p=p))
        w = np.exp(logw - logw.max()); w /= w.sum()
        probs = race_iid(P[:, F])                         # (M, k)
        L["bayes"] -= np.log(w @ probs[:, y])
        L["plugB"] -= np.log(race_iid((w @ P)[F])[y])
        SF = S[np.ix_(F, F)]
        C = np.linalg.cholesky(SF + 1e-12 * np.eye(k))
        L["lapmix"] -= np.log(race_probabilities(m[F], V=C, D=np.ones(k))[y])
        L["lapplug"] -= np.log(race_probabilities(m[F])[y])
        logw += np.log(probs[:, y])
        # Laplace step
        h = 1e-3; E = np.eye(k) * h
        f = lambda x: np.log(race_iid(x)[y])
        f0 = f(m[F]); g = np.array([(f(m[F] + E[i]) - f(m[F] - E[i])) / (2 * h) for i in range(k)])
        H = np.array([[(f(m[F] + E[i] + E[j]) - f(m[F] + E[i] - E[j]) - f(m[F] - E[i] + E[j]) + f(m[F] - E[i] - E[j])) / (4 * h * h)
                       for j in range(k)] for i in range(k)])
        w_, U = np.linalg.eigh(-0.5 * (H + H.T)); A = (U * np.maximum(w_, 0)) @ U.T
        K = np.linalg.solve(np.eye(k) + A @ SF, A)
        S = S - S[:, F] @ K @ S[:, F].T
        m = m + S[:, F] @ g
    ess = 1.0 / np.sum((np.exp(logw - logw.max()) / np.exp(logw - logw.max()).sum()) ** 2)
    return L, ess


if __name__ == "__main__":
    tot = {}
    for seed in range(30):
        L, ess = run(seed)
        print(seed, {k: round(v, 2) for k, v in L.items()}, "ESS", round(ess))
        for k_, v in L.items():
            tot.setdefault(k_, []).append(v)
    base = np.array(tot["bayes"])
    for k_, v in tot.items():
        d = np.array(v) - base
        print(k_, "minus bayes: mean %.3f se %.3f" % (d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
