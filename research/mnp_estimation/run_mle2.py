"""Round 2: isolate the simulation penalty. T = 50,000 choices (so
statistical error is small and simulation bias binds), known Sigma,
estimate mu only, n = 30, scored on identified alternatives (p > 0.01).

Fifth review: the saturated likelihood has NO finite maximizer when any
count is zero, and two of these eight replications (seeds 103 and 105)
have one -- the smallest alternative has p = 2.5e-5, so its expected
count is 1.27 and it is empty with probability e^-1.27 = 0.28. The
estimand is therefore the penalized (MAP) maximizer under a Jeffreys
Dirichlet prior, counts + 1/2, applied identically to every arm so the
comparison stays like-for-like. ALPHA = 0 reproduces the original
unpenalized run, which is well defined only on the six replications
with full support.
"""
import time
import numpy as np
from scipy.optimize import minimize
from winning.factor.races import race_probabilities
from winning.factor.polish import race_jacobian
import fastrace

n, T, REPS = 30, 50_000, 8
ALPHA = 0.5          # Jeffreys pseudocount; 0.0 = unpenalized (may not exist)
rng0 = np.random.default_rng(3)
mu_star = np.sort(rng0.normal(size=n)) * 0.8
mu_star -= mu_star.mean()
V = rng0.normal(size=(n, 2)) * 0.4
D = 0.6 + 0.5 * rng0.random(n)

def p_exact(mu):
    return race_probabilities(mu, V=V, D=D, points=257)

def negll_exact(mu_free, counts):
    mu = mu_free - mu_free.mean()
    return -counts @ np.log(np.maximum(p_exact(mu), 1e-300))

def grad_exact(mu_free, counts):
    mu = mu_free - mu_free.mean()
    p = np.maximum(p_exact(mu), 1e-300)
    J = race_jacobian(mu, V=V, D=D, points=257)
    g = -(J.T @ (counts / p))
    return g - g.mean()

def negll_ghk(mu_free, counts, R, seed):
    mu = mu_free - mu_free.mean()
    g = np.asarray(fastrace.ghk_all_shares(-mu, V, D, R, seed))
    g = np.maximum(g, 1e-12); g = g / g.sum()
    return -counts @ np.log(g)

def fd_grad(f, th, h=1e-4):
    g = np.zeros(len(th))
    for j in range(len(th)):
        e = np.zeros(len(th)); e[j] = h
        g[j] = (f(th + e) - f(th - e)) / (2 * h)
    return g

p_true = p_exact(mu_star)
ident = p_true > 0.01
print(f"identified alternatives: {ident.sum()}/{n}", flush=True)
for method in ("exact", "msl100", "msl1000"):
    errs, times = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(100 + rep)
        counts = rng.multinomial(T, p_true).astype(float) + ALPHA
        th0 = np.zeros(n)
        t0 = time.time()
        if method == "exact":
            res = minimize(negll_exact, th0, args=(counts,), jac=grad_exact,
                           method="L-BFGS-B", options={"maxiter": 150})
        else:
            R = 100 if method == "msl100" else 1000
            f = lambda th: negll_ghk(th, counts, R, 777)
            res = minimize(f, th0, jac=lambda th: fd_grad(f, th),
                           method="L-BFGS-B", options={"maxiter": 150})
        times.append(time.time() - t0)
        mu_hat = res.x - res.x.mean()
        errs.append(np.sqrt(np.mean((mu_hat[ident] - mu_star[ident]) ** 2)))
    print(f"{method:8s} rmse(mu | identified) {np.median(errs):.4f}  "
          f"median fit {np.median(times):7.2f}s", flush=True)
