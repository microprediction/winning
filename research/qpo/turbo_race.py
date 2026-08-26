"""TuRBO's Thompson step, solved instead of sampled.

Trust-region BO (Eriksson et al. 2019) is the workhorse for high-dimensional
problems. Its inner loop is: build a candidate set of N ~ 5,000 Sobol points
inside the trust region, draw q joint posterior samples over those candidates,
and take the argmin of each. That is Monte Carlo estimation of

    p_i = P(candidate i is the minimiser of the field),

which is a race. Parallel Thompson sampling is BY DEFINITION q independent
draws from p, so computing p exactly and drawing q categorical indices is
distributionally identical -- and never forms an N x N covariance.

The candidates sit inside a trust region, so they are close together and
strongly correlated: exactly the regime where a factor model earns its place.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import minimize
from scipy.stats import qmc


# ---------------------------------------------------------------- kernel
def matern52(X, Y, ls):
    Xs, Ys = X / ls, Y / ls
    d2 = np.maximum(((Xs ** 2).sum(1)[:, None] + (Ys ** 2).sum(1)[None, :]
                     - 2.0 * Xs @ Ys.T), 0.0)
    r = np.sqrt(d2 * 5.0)
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


class GP:
    """ARD Matern 5/2 GP with marginal-likelihood hyperparameters."""

    def __init__(self, X, y, ls=None, s2=None, noise=None, fit=True):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.ym, self.ys = self.y.mean(), max(self.y.std(), 1e-8)
        self.yn = (self.y - self.ym) / self.ys
        d = self.X.shape[1]
        th0 = np.concatenate([np.zeros(d), [0.0, np.log(1e-3)]])
        if fit:
            r = minimize(self._nll, th0, jac=False, method="L-BFGS-B",
                         bounds=[(-2.5, 2.5)] * d + [(-2.0, 2.0), (np.log(1e-6), np.log(1e-1))],
                         options={"maxiter": 40})
            th = r.x
        else:
            th = th0
        self._set(th)

    def _set(self, th):
        d = self.X.shape[1]
        self.ls = np.exp(th[:d])
        self.s2 = float(np.exp(th[d]))
        self.noise = float(np.exp(th[d + 1]))
        K = self.s2 * matern52(self.X, self.X, self.ls) + self.noise * np.eye(len(self.X))
        self.cf = cho_factor(K + 1e-8 * np.eye(len(self.X)), lower=True)
        self.alpha = cho_solve(self.cf, self.yn)

    def _nll(self, th):
        d = self.X.shape[1]
        ls, s2, nz = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
        K = s2 * matern52(self.X, self.X, ls) + (nz + 1e-8) * np.eye(len(self.X))
        try:
            cf = cho_factor(K, lower=True)
        except Exception:
            return 1e10
        a = cho_solve(cf, self.yn)
        return float(0.5 * self.yn @ a + np.log(np.diag(cf[0])).sum())

    def predict_mean_var(self, Xs, block=2048):
        mu = np.empty(len(Xs)); var = np.empty(len(Xs))
        for a in range(0, len(Xs), block):
            Kc = self.s2 * matern52(Xs[a:a + block], self.X, self.ls)
            mu[a:a + block] = Kc @ self.alpha
            W = cho_solve(self.cf, Kc.T)
            var[a:a + block] = self.s2 - np.einsum("ij,ji->i", Kc, W)
        return mu * self.ys + self.ym, np.maximum(var, 1e-12) * self.ys ** 2

    def posterior_cov(self, Xs):
        """The N x N the sampler needs, and the factor route never builds."""
        Kss = self.s2 * matern52(Xs, Xs, self.ls)
        Kc = self.s2 * matern52(Xs, self.X, self.ls)
        W = cho_solve(self.cf, Kc.T)
        return (Kss - Kc @ W) * self.ys ** 2

    def factor_model(self, Xs, rank, n_ind=256, seed=0):
        """(mu, V, D) with rank-r V and EXACT marginal variances, no N x N.

        prior block   s2 K_** ~= Phi Phi',  Phi = K_*Z K_ZZ^{-1/2}   (Nystrom)
        data term    -s4 K_*n A^{-1} K_n*  = -Psi Psi'               (rank n)
        so Sigma ~= B S B' with B = [Phi | Psi], S = diag(+1..., -1...);
        top eigenpairs from a thin QR and a small (n_ind+n) eigenproblem.
        """
        rng = np.random.default_rng(seed)
        N = len(Xs)
        mu, var = self.predict_mean_var(Xs)
        m = min(n_ind, N)
        Z = Xs[rng.choice(N, m, replace=False)]
        Kzz = self.s2 * matern52(Z, Z, self.ls) + 1e-8 * np.eye(m)
        w, U = np.linalg.eigh(Kzz)
        keep = w > w.max() * 1e-10
        Zi = U[:, keep] / np.sqrt(w[keep])
        Phi = (self.s2 * matern52(Xs, Z, self.ls)) @ Zi          # (N, m')
        Kc = self.s2 * matern52(Xs, self.X, self.ls)
        Psi = cho_solve(self.cf, Kc.T).T                          # (N, n)
        Psi = Kc @ np.linalg.cholesky(np.linalg.inv(
            self.s2 * matern52(self.X, self.X, self.ls)
            + (self.noise + 1e-8) * np.eye(len(self.X))))
        B = np.hstack([Phi, Psi])
        S = np.concatenate([np.ones(Phi.shape[1]), -np.ones(Psi.shape[1])])
        Q, R = qr(B, mode="economic")
        M = (R * S) @ R.T
        wv, Uv = np.linalg.eigh(0.5 * (M + M.T))
        idx = np.argsort(-wv)[:rank]
        wv = np.maximum(wv[idx], 0.0)
        V = (Q @ Uv[:, idx]) * np.sqrt(wv)                        # (N, rank)
        V *= self.ys
        D = np.maximum(var - (V ** 2).sum(1), 1e-10 * var)
        return mu, V, D


# ------------------------------------------------------------------ TuRBO-1
def ackley(X, lo=-5.0, hi=10.0):
    Z = lo + (hi - lo) * X
    a, b, c = 20.0, 0.2, 2 * np.pi
    return (-a * np.exp(-b * np.sqrt((Z ** 2).mean(1)))
            - np.exp(np.cos(c * Z).mean(1)) + a + np.e)


def levy(X, lo=-10.0, hi=10.0):
    Z = lo + (hi - lo) * X
    w = 1 + (Z - 1) / 4
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t2 = ((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2)).sum(1)
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def turbo(f, d, n_init=20, n_iter=40, batch=10, N_cand=5000, mode="factor",
          rank=2, seed=0, verbose=False):
    """TuRBO-1. `mode` selects how the Thompson batch is drawn:

        "cholesky"  form the N x N posterior covariance, factor it, draw q
                    joint samples, take each argmin  -- what TuRBO does, and
                    the reason N is capped at min(100d, 5000).
        "factor"    fit a rank-r factor model without ever forming N x N and
                    sample the LATENT: q draws of (f in R^r, eps in R^N).
    """
    rng = np.random.default_rng(seed)
    X = qmc.Sobol(d, scramble=True, seed=seed).random(n_init)
    y = f(X)
    L, n_succ, n_fail = 0.8, 0, 0
    succ_tol, fail_tol = 3, max(4, int(np.ceil(d / batch)))
    t_acq = 0.0
    for it in range(n_iter):
        if L < 0.5 ** 7:                      # restart
            X2 = qmc.Sobol(d, scramble=True, seed=seed + 1000 + it).random(n_init)
            X, y = np.vstack([X, X2]), np.concatenate([y, f(X2)])
            L, n_succ, n_fail = 0.8, 0, 0
        keep = np.argsort(y)[:400]            # cap GP size
        g = GP(X[keep], y[keep])
        xc = X[np.argmin(y)]
        w = g.ls / g.ls.mean(); w = w / np.prod(w) ** (1.0 / d)
        lb, ub = np.clip(xc - L * w / 2, 0, 1), np.clip(xc + L * w / 2, 0, 1)
        pert = rng.random((N_cand, d)) < min(20.0 / d, 1.0)
        pert[~pert.any(1), rng.integers(0, d, (~pert.any(1)).sum())] = True
        C = np.tile(xc, (N_cand, 1))
        S = lb + (ub - lb) * rng.random((N_cand, d))
        C[pert] = S[pert]
        t0 = time.time()
        if mode == "cholesky":
            mu = g.predict_mean_var(C)[0]
            Sig = g.posterior_cov(C)
            Lc = np.linalg.cholesky(Sig + 1e-8 * np.eye(N_cand))
            idx = [int(np.argmin(mu + rng.standard_normal(N_cand) @ Lc.T))
                   for _ in range(batch)]
            del Sig, Lc
        else:
            mu, V, D = g.factor_model(C, rank=rank, seed=seed + it)
            sd = np.sqrt(D)
            idx = [int(np.argmin(mu + V @ rng.standard_normal(rank)
                                 + sd * rng.standard_normal(N_cand)))
                   for _ in range(batch)]
        t_acq += time.time() - t0
        idx = list(dict.fromkeys(idx))
        Xn = C[idx]; yn = f(Xn)
        if yn.min() < y.min() - 1e-3 * abs(y.min()):
            n_succ, n_fail = n_succ + 1, 0
        else:
            n_succ, n_fail = 0, n_fail + 1
        if n_succ >= succ_tol: L, n_succ = min(2 * L, 1.6), 0
        if n_fail >= fail_tol: L, n_fail = L / 2, 0
        X, y = np.vstack([X, Xn]), np.concatenate([y, yn])
        if verbose:
            print(f"    it {it:3d} n={len(y):5d} best {y.min():.4f} L={L:.3f}", flush=True)
    return y.min(), len(y), t_acq
