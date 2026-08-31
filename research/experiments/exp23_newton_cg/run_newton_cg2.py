"""Experiment 23b: the Newton-CG rematch, with the handicaps removed.

The first attempt (run_newton_cg.py) lost by two orders of magnitude, and
its README diagnosed tail directions -- but the implementation also ran
UNpreconditioned CG (despite a docstring claiming otherwise) on RAW share
residuals, forfeiting exactly the log-domain scaling that makes the
production Jacobi solver work. This is the fair rematch: Newton on the
log-residual system, CG diagonally preconditioned by the same own-slopes
Jacobi uses, few CG steps per Newton step, step clip as in production.

Run: python run_newton_cg2.py
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import (abilities_from_probabilities_factor, hermite_nodes,
                      jacobian_vector_product, win_probabilities_factor)
from run_ghk_benchmark import make_problem, mc_shares

SEED = 21

def newton_cg_log(target, V, D, F, W, tol=1e-6, max_newton=30, max_cg=6,
                  cg_tol=0.1):
    """Newton on the LOG residual, diagonally preconditioned CG.

    System: d(log p)/dmu = diag(1/p) J, so solve M delta = r_log with
    M = diag(1/p) J, r_log = log p - log target, in the mean-zero gauge.
    M is not symmetric; run CG on the symmetrized normal-free form by
    preconditioning J itself: solve J delta = diag(p) r_log (back in
    share units but with the LOG residual driving), CG preconditioned by
    the own-slopes s_i (log-domain slope floors as in production).
    """
    n = len(target)
    logt = np.log(target)
    sd_tot2 = D + np.sum(np.atleast_2d(V)**2, axis=1)
    mu = abilities_from_probabilities_factor(
        target, np.zeros((n, 1)), sd_tot2, np.zeros((1, 1)), np.ones(1))
    n_jvp = 0
    prev = np.inf
    live = target > 1e-4 / n
    for it in range(max_newton):
        p = win_probabilities_factor(mu, V, D, F, W)
        # own-slopes for the preconditioner via the package front door,
        # which returns them from the same pass
        from winning.factor.races import race_probabilities
        _, slope = race_probabilities(mu, V=V, D=D, F=F, W=W,
                                      return_slopes=True)
        r_log = np.log(np.maximum(p, 1e-300)) - logt
        res = np.abs(r_log)[live].max()
        if res < tol:
            return mu, {"newton_iters": it, "jvp_calls": n_jvp,
                        "residual": res, "converged": True}
        # rhs in share units, driven by the log residual
        b = p * r_log
        b = b - b.mean()
        # diagonal preconditioner: production's own-slope magnitudes
        Mdiag = np.maximum(-slope, 1e-6 * np.max(-slope))
        delta = np.zeros(n)
        cr = b.copy()
        z = cr / Mdiag
        d = z.copy()
        rz = cr @ z
        for _ in range(max_cg):
            Ad = -jacobian_vector_product(mu, V, D, F, W, d, form="grid")
            Ad -= Ad.mean()
            n_jvp += 1
            alpha = rz / max(d @ Ad, 1e-300)
            delta += alpha * d
            cr -= alpha * Ad
            if np.linalg.norm(cr) < cg_tol * np.linalg.norm(b):
                break
            z = cr / Mdiag
            rz_new = cr @ z
            d = z + (rz_new / rz) * d
            rz = rz_new
        step = 1.0 if res <= prev * 1.2 else 0.5
        prev = res
        # production-style per-coordinate clip in LOG units:
        # delta is in mu units; residual-proportional cap as production
        lim = np.minimum(2.0, 10.0 * np.abs(r_log) + 0.05)
        mu = mu + step * np.clip(delta, -lim, lim)   # J neg-def: +delta descends
        mu -= mu.mean()
    return mu, {"newton_iters": max_newton, "jvp_calls": n_jvp,
                "residual": res, "converged": False}

def main():
    rng = np.random.default_rng(SEED)
    F, W = hermite_nodes(2)
    n = 200
    mu0, V, D = make_problem(n, 2, rng, spread=1.5)
    mu0 -= mu0.mean()
    target = mc_shares(mu0, V, D, 5_000_000, seed=61)
    target = np.maximum(target, 1e-9); target /= target.sum()

    t0 = time.perf_counter()
    _, ji = abilities_from_probabilities_factor(target, V, D, F, W,
                                                return_info=True)
    t_j = time.perf_counter() - t0
    print(f"jacobi      {t_j:7.1f}s  iters {ji['iterations']:3d}  "
          f"residual {ji['residual']:.1e}  converged {ji['converged']}")

    t0 = time.perf_counter()
    _, nc = newton_cg_log(target, V, D, F, W)
    t_n = time.perf_counter() - t0
    print(f"newton-cg2  {t_n:7.1f}s  newton {nc['newton_iters']:3d} / "
          f"jvp {nc['jvp_calls']:3d}  residual {nc['residual']:.1e}  "
          f"converged {nc['converged']}")

if __name__ == "__main__":
    main()
