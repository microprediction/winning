"""Experiment 23d: does the round-3 comparison move with field size?

Answers a reader question on the FAQ: production Jacobi's sweep count is
set by a spectral gap, so if that gap doesn't shrink with n while a JVP
costs about the same as a sweep, the wall-clock verdict should flip well
before n reaches a million. Three measurements, all at the "easy (exp23
shape)" problem from run_newton_cg3.py unless noted:

1. Per-call cost: one IBP-form JVP against one forward pass (with
   slopes), n=200 to n=12,800.
2. Production Jacobi's sweep count to reach tol=1e-8, same n range.
3. The round-3 hybrid solver itself, unmodified, at n=3,200 (exp23 only
   ever ran round 3 at n=200) -- does its Newton/JVP count also stay
   flat, or does the unmodified formulation degrade before field size
   gets a chance to flip anything?

Run: python run_field_size_scaling.py
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.core import jacobian_vector_product, hermite_nodes
from run_newton_cg3 import hybrid, trial

F, W = hermite_nodes(2)


def easy_problem(n):
    rng = np.random.default_rng(21)
    mu = np.sort(rng.normal(size=n)) * 1.5
    mu -= mu.mean()
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.6 + 0.5 * rng.random(n)
    return mu, V, D


def bench(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def part1_per_call_cost():
    print("\n1. Per-call cost: JVP vs. forward pass\n")
    for n, reps in [(200, 20), (800, 8), (3200, 3), (12800, 1)]:
        mu, V, D = easy_problem(n)
        h = np.random.default_rng(0).normal(size=n)
        fwd_time = bench(lambda: race_probabilities(mu, V=V, D=D, F=F, W=W,
                                                      return_slopes=True), reps)
        jvp_time = bench(lambda: jacobian_vector_product(mu, V, D, F, W, h,
                                                           form="ibp"), reps)
        print(f"  n={n:6d}  forward={fwd_time*1000:9.2f} ms  "
              f"jvp={jvp_time*1000:9.2f} ms  ratio={jvp_time/fwd_time:5.2f}x")


def part2_sweep_count():
    print("\n2. Production Jacobi sweep count vs. field size\n")
    for n in (200, 800, 3200, 12800):
        mu, V, D = easy_problem(n)
        p_target = race_probabilities(mu, V=V, D=D, F=F, W=W)
        t0 = time.perf_counter()
        _, info = abilities_from_race(p_target, V=V, D=D, F=F, W=W,
                                       return_info=True, tol=1e-8)
        dt = time.perf_counter() - t0
        print(f"  n={n:6d}  sweeps={info['iterations']:3d}  time={dt:7.2f}s  "
              f"res={info['max_log_residual']:.1e}")


def part3_hybrid_at_scale():
    print("\n3. The round-3 hybrid, unmodified, at n=3,200\n")
    for n in (200, 3200):
        mu, V, D = easy_problem(n)
        trial(f"  easy n={n}", mu, V, D, F, W)


if __name__ == "__main__":
    part1_per_call_cost()
    part2_sweep_count()
    part3_hybrid_at_scale()
