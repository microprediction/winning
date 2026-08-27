"""Covariate MNP estimation: the referee's experiment.

Observation-specific design matrices X_t (n x d), utilities mu_t = X_t
beta, one choice per observation, known rank-2 factor covariance.
Estimate beta (d = 3) by individual-level maximum likelihood:

  exact : lattice p_t and Jacobian J_t per observation; score
          sum_t X_t' J_t[y_t,:] / p_{t,y_t}   (analytic)
  MSL-R : GHK probabilities (rust, full vector supplied at one-call
          price, CRN across iterations), finite-difference gradient

Reported with across-replication standard errors.
"""
import time
import numpy as np
from scipy.optimize import minimize
from winning.factor.races import race_probabilities
from winning.factor.polish import race_jacobian
import fastrace

n, d, T, REPS = 10, 3, 800, 8
rng0 = np.random.default_rng(9)
beta_star = np.array([0.8, -0.5, 0.3])
V = rng0.normal(size=(n, 2)) * 0.4
D = 0.6 + 0.5 * rng0.random(n)

def simulate(rep):
    rng = np.random.default_rng(1000 + rep)
    X = rng.normal(size=(T, n, d)) * 0.7
    y = np.empty(T, np.int64)
    L = np.linalg.cholesky(V @ V.T + np.diag(D))
    for t in range(T):
        mu_t = X[t] @ beta_star
        y[t] = np.argmin(mu_t + L @ rng.standard_normal(n))
    return X, y

def negll_exact(beta, X, y, want_grad):
    ll = 0.0
    g = np.zeros(d)
    for t in range(len(y)):
        mu_t = X[t] @ beta
        mu_t = mu_t - mu_t.mean()
        p = np.maximum(race_probabilities(mu_t, V=V, D=D, points=161), 1e-300)
        ll -= np.log(p[y[t]])
        if want_grad:
            J = race_jacobian(mu_t, V=V, D=D, points=161)
            gmu = -J[y[t]] / p[y[t]]
            gmu = gmu - gmu.mean()
            g += X[t].T @ gmu
    return (ll, g) if want_grad else ll

def negll_ghk(beta, X, y, R, seed):
    ll = 0.0
    for t in range(len(y)):
        mu_t = X[t] @ beta
        p = np.asarray(fastrace.ghk_all_shares(-(mu_t - mu_t.mean()), V, D, R, seed))
        p = np.maximum(p, 1e-12); p = p / p.sum()
        ll -= np.log(p[y[t]])
    return ll

for method in ("exact", "msl100", "msl1000"):
    errs, times = [], []
    for rep in range(REPS):
        X, y = simulate(rep)
        t0 = time.time()
        if method == "exact":
            res = minimize(lambda b: negll_exact(b, X, y, True), np.zeros(d),
                           jac=True, method="L-BFGS-B",
                           options={"maxiter": 60})
        else:
            R = 100 if method == "msl100" else 1000
            f = lambda b: negll_ghk(b, X, y, R, 777)
            def fd(b, h=1e-4):
                g = np.zeros(d)
                for j in range(d):
                    e = np.zeros(d); e[j] = h
                    g[j] = (f(b + e) - f(b - e)) / (2 * h)
                return g
            res = minimize(f, np.zeros(d), jac=fd, method="L-BFGS-B",
                           options={"maxiter": 60})
        times.append(time.time() - t0)
        errs.append(np.sqrt(np.mean((res.x - beta_star) ** 2)))
    errs = np.array(errs)
    print(f"{method:8s} rmse(beta) {errs.mean():.4f} +- {errs.std(ddof=1)/np.sqrt(REPS):.4f}"
          f"  median fit {np.median(times):7.1f}s", flush=True)
