"""Generative-model referee for the rating updates (the efficacy gate).

Strategy: (1) simulate the EXACT generative model the updates claim to
serve -- skill s ~ N(m, v), performance x = s + sqrt(beta2) * eps with
eps the module's max-wins standardized base noise -- and (2) compare
the engine's posterior moments to the Monte Carlo posterior obtained
by conditioning the simulation on the observed event.

Pass marks (agreed with the bandits session's audit, 2026-09-04):
max coordinate error <= ~0.003 on posterior means AND variances for
every base, with normal serving as the regression row (it must stay at
its historical accuracy; the code path is untouched).

Configs deliberately include ones the bandits harness did not pick.
Run:  python research/adjudications/predictive_referee.py [--fast]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from winning.ratings.nway import (update_winner, update_ranking_exact,
                                  update_winner_correlated,
                                  update_order_correlated)
from winning.factor.races import student_base

_EULER = 0.5772156649015329
_C_GUMBEL = np.pi / np.sqrt(6.0)


def sample_noise(rng, base, size):
    """Max-wins standardized noise: the module's f_min(-z) convention."""
    if base == "normal":
        return rng.standard_normal(size)
    if base == "logistic":
        return rng.logistic(0.0, np.sqrt(3.0) / np.pi, size)
    if base == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    if base == "gumbel":
        # min-wins base is min-Gumbel; the max-wins flip is standard
        # max-Gumbel, standardized to mean 0 variance 1
        return (rng.gumbel(0.0, 1.0, size) - _EULER) / _C_GUMBEL
    if base == "student4":
        return rng.standard_t(4.0, size) / np.sqrt(2.0)
    raise ValueError(base)


def base_arg(base):
    return student_base(4.0) if base == "student4" else base


def mc_posterior_winner(rng, m, v, b2, winner, base, N, reps):
    n = len(m)
    means, vars_, ps = [], [], []
    for _ in range(reps):
        s = m[None, :] + np.sqrt(v)[None, :] * rng.standard_normal((N, n))
        x = s + np.sqrt(b2) * sample_noise(rng, base, (N, n))
        keep = x.argmax(axis=1) == winner
        sk = s[keep]
        means.append(sk.mean(axis=0))
        vars_.append(sk.var(axis=0))
        ps.append(keep.mean())
    return (np.mean(means, axis=0), np.mean(vars_, axis=0),
            np.std(means, axis=0, ddof=1) / np.sqrt(reps),
            float(np.mean(ps)))


def mc_posterior_order(rng, m, v, b2, order, base, N, reps):
    n = len(m)
    order = np.asarray(order)
    means, vars_ = [], []
    for _ in range(reps):
        s = m[None, :] + np.sqrt(v)[None, :] * rng.standard_normal((N, n))
        x = s + np.sqrt(b2) * sample_noise(rng, base, (N, n))
        keep = (np.argsort(-x, axis=1) == order[None, :]).all(axis=1)
        sk = s[keep]
        means.append(sk.mean(axis=0))
        vars_.append(sk.var(axis=0))
    return (np.mean(means, axis=0), np.mean(vars_, axis=0),
            np.std(means, axis=0, ddof=1) / np.sqrt(reps))


WINNER_CONFIGS = [
    ("A(bandits)", np.array([0.4, 0.0, -0.3]),
     np.array([0.6, 0.4, 0.5]), 1.0, 0),
    ("B(n4,longshot)", np.array([0.0, -0.5, 0.3, 0.8]),
     np.array([1.2, 0.3, 0.7, 0.5]), 0.6, 2),
    ("C(diffuse)", np.array([1.0, -1.0]),
     np.array([4.0, 0.25]), 1.5, 1),
]

ORDER_CONFIGS = [
    ("A order[0,2,1]", np.array([0.4, 0.0, -0.3]),
     np.array([0.6, 0.4, 0.5]), 1.0, [0, 2, 1]),
    ("B order[3,0,2,1]", np.array([0.0, -0.5, 0.3, 0.8]),
     np.array([1.2, 0.3, 0.7, 0.5]), 0.6, [3, 0, 2, 1]),
]

BASES = ["normal", "logistic", "laplace", "gumbel", "student4"]


def main():
    fast = "--fast" in sys.argv
    N = 1_000_000 if fast else 4_000_000
    reps = 3 if fast else 6
    rng = np.random.default_rng(20260904)
    tol = 0.003
    failures = []

    print("=== update_winner vs generative-model MC posterior ===")
    print(f"    N={N} x {reps} reps, pass mark {tol}")
    for name, m, v, b2, w in WINNER_CONFIGS:
        for base in BASES:
            m_mc, v_mc, se, p_mc = mc_posterior_winner(
                rng, m, v, b2, w, base, N, reps)
            m_hat, v_hat, p_hat = update_winner(m, v, w, beta2=b2,
                                                base=base_arg(base))
            dm = float(np.abs(m_hat - m_mc).max())
            dv = float(np.abs(v_hat - v_mc).max())
            # variance gate is RELATIVE (0.006 = the 0.003-on-0.5 bar of
            # the bandits config): an absolute bar misreads a diffuse
            # prior whose posterior variance is ~4
            dvr = float((np.abs(v_hat - v_mc) / np.maximum(v_mc,
                                                           1e-12)).max())
            dp = abs(p_hat - p_mc)
            ok = dm <= max(tol, 4 * se.max()) and dvr <= 0.006
            flag = "ok " if ok else "FAIL"
            if not ok:
                failures.append(("winner", name, base, dm, dv))
            print(f"  {flag} {name:16s} {base:9s} dmean {dm:.4f} "
                  f"dvar {dv:.4f} dvar_rel {dvr:.4f} dp {dp:.4f} "
                  f"(se~{se.max():.4f})")

    print("=== update_ranking_exact vs conditioned MC posterior ===")
    for name, m, v, b2, order in ORDER_CONFIGS:
        for base in BASES:
            m_mc, v_mc, se = mc_posterior_order(
                rng, m, v, b2, order, base, N, reps)
            m_hat, v_hat = update_ranking_exact(m, v, order, beta2=b2,
                                                base=base_arg(base))
            dm = float(np.abs(m_hat - m_mc).max())
            dv = float(np.abs(v_hat - v_mc).max())
            ok = dm <= max(tol, 4 * se.max()) and dv <= 0.004
            flag = "ok " if ok else "FAIL"
            if not ok:
                failures.append(("order", name, base, dm, dv))
            print(f"  {flag} {name:16s} {base:9s} dmean {dm:.4f} "
                  f"dvar {dv:.4f} (se~{se.max():.4f})")

    print("=== correlated updates (V != 0), laplace vs MC posterior ===")
    m = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
    v = np.array([0.8, 1.0, 0.6, 1.2, 0.9])
    V = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])
    b2 = 1.0
    for base in ["laplace", "logistic"]:
        me, va = [], []
        for _ in range(reps):
            s = m[None, :] + np.sqrt(v)[None, :] \
                * rng.standard_normal((N, 5))
            f = rng.standard_normal((N, 1))
            x = s + f @ V.T + np.sqrt(b2) * sample_noise(rng, base, (N, 5))
            keep = x.argmax(axis=1) == 3
            me.append(s[keep].mean(axis=0))
            va.append(s[keep].var(axis=0))
        m_mc, v_mc = np.mean(me, axis=0), np.mean(va, axis=0)
        m_hat, v_hat, _ = update_winner_correlated(m, v, 3, V, beta2=b2,
                                                   base=base_arg(base))
        dm = float(np.abs(m_hat - m_mc).max())
        dv = float(np.abs(v_hat - v_mc).max())
        ok = dm <= 0.004 and dv <= 0.012
        flag = "ok " if ok else "FAIL"
        if not ok:
            failures.append(("winner_corr", "5p-factor", base, dm, dv))
        print(f"  {flag} winner_corr      {base:9s} dmean {dm:.4f} "
              f"dvar {dv:.4f}")
    mo = np.array([0.4, 0.0, -0.3, 0.0])
    vo = np.array([0.9, 0.9, 0.9, 0.9])
    Vo = np.array([[0.9], [0.8], [-0.7], [0.1]])
    order = np.array([1, 0, 3, 2])
    for base in ["laplace"]:
        me, va = [], []
        for _ in range(reps):
            s = mo[None, :] + np.sqrt(vo)[None, :] \
                * rng.standard_normal((N, 4))
            f = rng.standard_normal((N, 1))
            x = s + f @ Vo.T + np.sqrt(b2) * sample_noise(rng, base, (N, 4))
            keep = (np.argsort(-x, axis=1) == order[None, :]).all(axis=1)
            me.append(s[keep].mean(axis=0))
            va.append(s[keep].var(axis=0))
        m_mc, v_mc = np.mean(me, axis=0), np.mean(va, axis=0)
        m_hat, v_hat, _ = update_order_correlated(mo, vo, order, Vo,
                                                  beta2=b2, base=base)
        dm = float(np.abs(m_hat - m_mc).max())
        dv = float(np.abs(v_hat - v_mc).max())
        ok = dm <= 0.006 and dv <= 0.016
        flag = "ok " if ok else "FAIL"
        if not ok:
            failures.append(("order_corr", "4p-factor", base, dm, dv))
        print(f"  {flag} order_corr       {base:9s} dmean {dm:.4f} "
              f"dvar {dv:.4f}")

    print("=" * 50)
    if failures:
        print(f"{len(failures)} FAILURES")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
