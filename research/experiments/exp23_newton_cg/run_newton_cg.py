"""Experiment 23: Newton-CG calibration via the Jacobian-vector product.

The sixth review asked that the SPD Laplacian/JVP machinery either be
demonstrated in the solver or explicitly labeled future-capable. This
demonstrates it: calibration by damped Newton-CG on the mean-zero quotient,
with each Newton system B'JB delta = B'r solved matrix-free by conjugate
gradients using the exact-grid JVP (form="grid"), Jacobi (own-slope)
preconditioned. Compared against the default damped Jacobi iteration on the
same problems.

Run:  python experiments/exp23_newton_cg/run_newton_cg.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    jacobian_vector_product,
    win_probabilities_factor,
)
from run_ghk_benchmark import make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def newton_cg_calibrate(target, V, D, F, W, tol=1e-6, max_newton=25,
                        cg_tol=0.1, max_cg=8):
    """Damped Newton-CG on the quotient: matrix-free CG on J delta = r with
    mean-projection enforcing the quotient, halved steps on residual
    increase."""
    n = len(target)
    logp = np.log(target)
    # same warm start as the Jacobi solver: exact independent inverse with
    # total variances (cold starts stall: the share-scale residual
    # under-weights small-share coordinates far from the solution)
    sd_tot2 = D + np.sum(np.atleast_2d(V)**2, axis=1)
    mu = abilities_from_probabilities_factor(
        target, np.zeros((n, 1)), sd_tot2, np.zeros((1, 1)), np.ones(1))
    n_jvp = 0
    prev = np.inf
    for it in range(max_newton):
        p = win_probabilities_factor(mu, V, D, F, W)
        r = p - target
        res = np.abs(np.log(np.maximum(p, 1e-300)) - logp)[target > 1e-4 / n].max()
        if res < tol:
            return mu, {"newton_iters": it, "jvp_calls": n_jvp,
                        "residual": res, "converged": True}
        # Newton step delta = J^{-1} r. Min-wins J is MINUS a weighted
        # Laplacian (negative definite on the quotient), so run CG on the
        # SPD operator A = -J with right-hand side -r: A delta = -r.
        delta = np.zeros(n)
        b = -(r - r.mean())
        cr = b.copy()
        d = cr.copy()
        rs = cr @ cr
        for _ in range(max_cg):
            Ad = -jacobian_vector_product(mu, V, D, F, W, d, form="grid")
            Ad -= Ad.mean()
            n_jvp += 1
            alpha = rs / max(d @ Ad, 1e-300)
            delta += alpha * d
            cr -= alpha * Ad
            rs_new = cr @ cr
            if np.sqrt(rs_new) < cg_tol * np.sqrt(b @ b):
                break
            d = cr + (rs_new / rs) * d
            rs = rs_new
        step = 1.0
        if res > prev * 1.2:
            step = 0.5
        prev = res
        mu = mu - step * np.clip(delta, -2, 2)
        mu -= mu.mean()
    return mu, {"newton_iters": max_newton, "jvp_calls": n_jvp,
                "residual": res, "converged": False}


def main():
    rng = np.random.default_rng(SEED)
    rows = ["N,method,seconds,iters_or_jvps,residual,converged"]
    F, W = hermite_nodes(2)
    for n in (200, 1000):
        mu, V, D = make_problem(n, 2, rng, spread=1.5)
        mu -= mu.mean()
        target = mc_shares(mu, V, D, 5_000_000, seed=61)
        target = np.maximum(target, 1e-9); target /= target.sum()

        t0 = time.perf_counter()
        _, ji = abilities_from_probabilities_factor(
            target, V, D, F, W, return_info=True)
        t_j = time.perf_counter() - t0
        print(f"N={n} Jacobi:    {t_j:6.0f}s, {ji['iterations']} iterations, "
              f"residual {ji['residual']:.1e}, converged {ji['converged']}")
        rows.append(f"{n},jacobi,{t_j:.1f},{ji['iterations']},"
                    f"{ji['residual']:.3e},{ji['converged']}")

        t0 = time.perf_counter()
        _, nc = newton_cg_calibrate(target, V, D, F, W)
        t_n = time.perf_counter() - t0
        print(f"N={n} Newton-CG: {t_n:6.0f}s, {nc['newton_iters']} Newton / "
              f"{nc['jvp_calls']} JVPs, residual {nc['residual']:.1e}, "
              f"converged {nc['converged']}")
        rows.append(f"{n},newton_cg,{t_n:.1f},"
                    f"{nc['newton_iters']}n+{nc['jvp_calls']}jvp,"
                    f"{nc['residual']:.3e},{nc['converged']}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
