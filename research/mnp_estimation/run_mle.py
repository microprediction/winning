"""MNP estimation head to head: exact-gradient MLE vs GHK-based MSL.

True model: n alternatives, utilities mu* (mean-zero) with factor
covariance c* V0 V0' + diag(D). T choices observed (multinomial counts
under p(theta*)). Estimate theta = (mu mean-zero, c) by:

  exact : lattice probabilities + analytic score  J'(counts/p)
  MSL-R : same likelihood with GHK probabilities at R draws, common
          random numbers across iterations (standard practice), and
          finite-difference gradients on the smoothed simulator

Scored on parameter RMSE and wall clock over replications; also the
log-likelihood gap at the truth, which isolates the MSL bias
E[log p_hat] < log p.
"""
import time
import numpy as np
from scipy.optimize import minimize
from winning.factor.races import race_probabilities
from winning.factor.polish import race_jacobian
import fastrace

n, T, REPS = 12, 2000, 20
rng0 = np.random.default_rng(3)
mu_star = np.sort(rng0.normal(size=n)) * 0.8
mu_star -= mu_star.mean()
V0 = rng0.normal(size=(n, 1)) * 0.5
D = 0.6 + 0.5 * rng0.random(n)
c_star = 1.0

def p_exact(mu, c):
    return race_probabilities(mu, V=c * V0, D=D, points=257)

def negll_exact(theta, counts):
    mu = theta[:n] - theta[:n].mean()
    c = theta[n]
    p = np.maximum(p_exact(mu, c), 1e-300)
    return -counts @ np.log(p)

def grad_exact(theta, counts):
    mu = theta[:n] - theta[:n].mean()
    c = theta[n]
    p = np.maximum(p_exact(mu, c), 1e-300)
    J = race_jacobian(mu, V=c * V0, D=D, points=257)
    g_mu = -(J.T @ (counts / p))
    g_mu -= g_mu.mean()
    h = 1e-4
    gc = -(counts @ (np.log(np.maximum(p_exact(mu, c + h), 1e-300))
                     - np.log(np.maximum(p_exact(mu, c - h), 1e-300)))) / (2 * h)
    return np.concatenate([g_mu, [gc]])

def p_ghk(mu, c, R, seed):
    g = np.asarray(fastrace.ghk_all_shares(-mu, c * V0, D, R, seed))
    g = np.maximum(g, 1e-12)
    return g / g.sum()

def negll_ghk(theta, counts, R, seed):
    mu = theta[:n] - theta[:n].mean()
    return -counts @ np.log(p_ghk(mu, theta[n], R, seed))

def fd_grad(f, theta, h=1e-4):
    g = np.zeros(len(theta))
    for j in range(len(theta)):
        e = np.zeros(len(theta)); e[j] = h
        g[j] = (f(theta + e) - f(theta - e)) / (2 * h)
    return g

p_true = p_exact(mu_star, c_star)
results = {}
for method in ("exact", "msl100", "msl1000"):
    errs_mu, errs_c, times, llgaps = [], [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        counts = rng.multinomial(T, p_true).astype(float)
        theta0 = np.concatenate([np.zeros(n), [0.8]])
        t0 = time.time()
        if method == "exact":
            res = minimize(negll_exact, theta0, args=(counts,),
                           jac=grad_exact, method="L-BFGS-B",
                           options={"maxiter": 200})
        else:
            R = 100 if method == "msl100" else 1000
            seed = 777           # CRN: same seed every evaluation
            f = lambda th: negll_ghk(th, counts, R, seed)
            res = minimize(f, theta0, jac=lambda th: fd_grad(f, th),
                           method="L-BFGS-B", options={"maxiter": 200})
        dt = time.time() - t0
        mu_hat = res.x[:n] - res.x[:n].mean()
        errs_mu.append(np.sqrt(np.mean((mu_hat - mu_star) ** 2)))
        errs_c.append(res.x[n] - c_star)
        times.append(dt)
        if method != "exact":
            R = 100 if method == "msl100" else 1000
            llgaps.append(negll_ghk(np.concatenate([mu_star, [c_star]]),
                                    counts, R, 777)
                          - negll_exact(np.concatenate([mu_star, [c_star]]),
                                        counts))
    results[method] = (errs_mu, errs_c, times, llgaps)
    line = (f"{method:8s} rmse(mu) {np.median(errs_mu):.4f}  "
            f"bias(c) {np.mean(errs_c):+.4f}  sd(c) {np.std(errs_c):.4f}  "
            f"median fit {np.median(times):6.2f}s")
    if llgaps:
        line += f"  MSL loglik bias at truth {np.mean(llgaps):+.2f} nats"
    print(line, flush=True)
