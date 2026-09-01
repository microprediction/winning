"""Experiment 23c: Newton-Krylov, correctly formulated.

The first two attempts lost for identifiable reasons: raw-share
residuals and unpreconditioned CG (23), then grid-form JVP without the
null projection or symmetrization (23b). This one follows the full
recipe: log-residual Newton system L delta = P g~, symmetrized as
(P^-1/2 L P^-1/2) y = P^1/2 g~ with delta = P^-1/2 y; the operator is
one IBP-form JVP per application (the explicitly symmetric Laplacian
form); PCG preconditioned by the own-log-slope diagonal -J_ii/p_i; the
null vector sqrt(p) projected out of rhs and iterates; a
residual-proportional trust clip; accept a step only if the true forward
log residual improves, else halve, else fall back to one production
Jacobi sweep. Warm start: four production Jacobi sweeps.

Run: python run_newton_cg3.py
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.core import jacobian_vector_product, hermite_nodes

def forward(mu, V, D, F, W):
    return race_probabilities(mu, V=V, D=D, F=F, W=W, return_slopes=True)

def jacobi_sweep(mu, p, slope, logt, alpha=1.0):
    r = np.log(np.maximum(p, 1e-300)) - logt
    d = np.minimum(slope / np.maximum(p, 1e-300), -1e-6)
    lim = np.minimum(2.0, 10.0 * np.abs(r))
    mu = mu - np.clip(alpha * r / d, -lim, lim)
    return mu - mu.mean()

def hybrid(target, V, D, F, W, tol=1e-8, warm=4, max_newton=10, max_cg=5):
    n = len(target)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    n_fwd = n_jvp = 0
    # warm start: production sweeps
    for _ in range(warm):
        p, slope = forward(mu, V, D, F, W); n_fwd += 1
        mu = jacobi_sweep(mu, p, slope, logt)
    for it in range(max_newton):
        p, slope = forward(mu, V, D, F, W); n_fwd += 1
        g = np.log(np.maximum(p, 1e-300)) - logt
        res = np.abs(g).max()
        if res < tol:
            return mu, dict(newton=it, fwd=n_fwd, jvp=n_jvp,
                            residual=res, converged=True)
        gt = g - float(p @ g)                 # project the common component
        sqrt_p = np.sqrt(np.maximum(p, 1e-300))
        b = sqrt_p * gt
        b = b - (b @ sqrt_p) * sqrt_p         # null-vector projection
        Mdiag = np.maximum(-slope / np.maximum(p, 1e-300), 1e-12)
        y = np.zeros(n); cr = b.copy()
        z = cr / Mdiag; d = z.copy(); rz = cr @ z
        for _ in range(max_cg):
            dm = d / sqrt_p
            Ad = -jacobian_vector_product(mu, V, D, F, W, dm,
                                          form="ibp") / sqrt_p
            Ad = Ad - (Ad @ sqrt_p) * sqrt_p
            n_jvp += 1
            denom = d @ Ad
            if denom <= 1e-300: break
            a = rz / denom
            y += a * d; cr -= a * Ad
            if np.linalg.norm(cr) < 0.1 * np.linalg.norm(b): break
            z = cr / Mdiag
            rz_new = cr @ z
            d = z + (rz_new / rz) * d
            rz = rz_new
        delta = y / sqrt_p
        delta = delta - delta.mean()
        lim = np.minimum(2.0, 10.0 * np.abs(g) + 0.02)   # trust clip
        delta = np.clip(delta, -lim, lim)
        # accept only on true improvement; halve once; else Jacobi fallback
        accepted = False
        for step in (1.0, 0.5):
            mu_try = mu + step * delta
            mu_try -= mu_try.mean()
            p2, s2 = forward(mu_try, V, D, F, W); n_fwd += 1
            res2 = np.abs(np.log(np.maximum(p2, 1e-300)) - logt).max()
            if res2 < res:
                mu, accepted = mu_try, True
                break
        if not accepted:
            mu = jacobi_sweep(mu, p, slope, logt)
    p, _ = forward(mu, V, D, F, W)
    res = np.abs(np.log(np.maximum(p, 1e-300)) - logt).max()
    return mu, dict(newton=max_newton, fwd=n_fwd, jvp=n_jvp,
                    residual=res, converged=res < tol)

def trial(name, mu0, V, D, F, W):
    p_target = race_probabilities(mu0, V=V, D=D, F=F, W=W)
    t0 = time.perf_counter()
    _, info = abilities_from_race(p_target, V=V, D=D, F=F, W=W,
                                  return_info=True, tol=1e-8)
    tj = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, h = hybrid(p_target, V, D, F, W)
    th = time.perf_counter() - t0
    print(f"{name:24s} jacobi {tj:6.1f}s it={info['iterations']:2d} "
          f"res={info['max_log_residual']:.1e} | hybrid {th:6.1f}s "
          f"newton={h['newton']} jvp={h['jvp']} res={h['residual']:.1e} "
          f"conv={h['converged']}")

def main():
    rng = np.random.default_rng(21)
    F, W = hermite_nodes(2)
    n = 200
    # easy: the exp23 problem shape
    mu = np.sort(rng.normal(size=n)) * 1.5; mu -= mu.mean()
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.6 + 0.5 * rng.random(n)
    trial("easy (exp23 shape)", mu, V, D, F, W)
    # hard: strong correlation, where coordinate updates zig-zag
    V2 = rng.normal(size=(n, 2)) * 1.6
    D2 = 0.15 + 0.1 * rng.random(n)
    trial("hard (strong correlation)", mu, V2, D2, F, W)

if __name__ == "__main__":
    main()
