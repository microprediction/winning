"""The bid solicitation problem: who to call, and what solving it properly buys.

The customer's problem from "Who Ya Gonna Call?" (Cotton): choose a subset V
of dealers to solicit, minimizing

    Psi(V) = s * E[min_{i in V} m_i]  +  sum_{i in V} I_i

where m_i is dealer i's stochastic markup, s the trade size and I_i the cost
of inquiring with dealer i (time, information leakage). Two claims deserve
quantification:

  1. Psi is submodular, so the minimization is tractable (Lovasz); but it is
     submodular MINIMIZATION, not monotone submodular maximization, so the
     textbook greedy (1-1/e) guarantee does not apply. How close is greedy,
     or greedy plus local search, to the exact optimum in realistic worlds?
  2. The practical incumbent policy ranks dealers by market share (win
     probability) and calls the top k. Market share measures how OFTEN a
     dealer is best, not by HOW MUCH; a wide-tailed opportunist with modest
     share can lower E[min] more than a consistent major. How many bps does
     share-ranking leave on the table?

World: 14 dealers on a lattice (exact enumeration of all 16,383 call lists
is feasible, so every policy is scored against the true optimum).
  - 5 majors:      loc ~ 0.0,  sd 1.0, Gaussian     (high share)
  - 5 regionals:   loc ~ 0.55, sd 1.0, Gaussian     (mid share)
  - 4 opportunists: loc 1.6, but with probability 0.05 an aggressive
    N(loc-4.5, 0.6) quote: rarely best, best by a lot (low share, high value)
E[min_V] computed exactly from lattice survival products (cross-checked
against thurstone.Race.winner_density). Market shares are the exact state
prices of the all-dealer race. The calibrated pipeline mimics practice:
observe shares only, assume a GAUSSIAN base (misspecified: truth has the
opportunist mixture), invert with thurstone.AbilityCalibrator, choose V on
the calibrated model, pay the true Psi.

Policies: exact (enumeration), greedy-add, greedy + add/drop/swap local
search, top-k by share (k chosen with an oracle for the incumbent's best k),
best random-k, call-everyone. Cost regimes: uniform I_i and heterogeneous
I_i; trade sizes s in {2, 5, 10}.

Run:  .venv/bin/python research/bid_solicitation.py

Measured (July 2026, 12 worlds x 2 cost regimes x 3 trade sizes; gaps as %
of achievable savings, i.e. best-single-dealer Psi minus exact-optimum Psi):
    submodularity: zero diminishing-returns violations in 500 random triples
    greedy = exact optimum in all 72 scenarios (0.00% gap; local search never
        needed) -- the "hard" combinatorial problem is empirically trivial
        once the objective is computed on the lattice
    top-k by market share (even with oracle k): leaves 3.0% (uniform costs)
        to 7.9% (heterogeneous) at s=2, worst world 15.1%; near-zero at s=10
        where the optimum calls nearly everyone anyway
    share + GAUSSIAN calibration (misspecified family): leaves 5.4-14.0% at
        s=2 -- getting the density family right matters more than the
        optimizer, echoing the F1 non-Gaussian finding
    call-everyone: 14-30% at s=2
    the optimum calls all 4 low-share opportunists at s>=5; top-k under-calls
        them in the binding regimes (2.0 vs 2.7 at heterogeneous s=2)
Interpretation: solicitation optimization matters exactly when inquiry costs
bind relative to trade size (small tickets, most trades). The loss from
naive policies is not the optimizer (greedy suffices) but the input model:
market share alone, pushed through a Gaussian assumption, undervalues
fat-tailed quote distributions -- dealers who rarely win but win by a lot.
"""

from __future__ import annotations

import itertools
import numpy as np
from thurstone import AbilityCalibrator, Density, Race, UniformLattice

L = 240
UNIT = 0.05
LAT = UniformLattice(L=L, unit=UNIT)
GRID = np.arange(-L, L + 1) * UNIT
N = 14
SEED = 7


def normal_pmf(loc, sd):
    p = np.exp(-0.5 * ((GRID - loc) / sd) ** 2)
    return p / p.sum()


def dealer_world(rng):
    """True markup densities, one per dealer, and their labels."""
    pmfs, labels = [], []
    for k in range(5):
        pmfs.append(normal_pmf(rng.normal(0.0, 0.15), 1.0))
        labels.append(f"major{k}")
    for k in range(5):
        pmfs.append(normal_pmf(0.55 + rng.normal(0.0, 0.15), 1.0))
        labels.append(f"regional{k}")
    for k in range(4):
        loc = 1.6 + rng.normal(0.0, 0.1)
        p = 0.95 * normal_pmf(loc, 1.0) + 0.05 * normal_pmf(loc - 4.5, 0.6)
        pmfs.append(p / p.sum())
        labels.append(f"opportunist{k}")
    return np.array(pmfs), labels


def survival(pmfs):
    """S[i, k] = P(m_i > x_k), lattice convention: mass at grid points."""
    return 1.0 - np.cumsum(pmfs, axis=1)


def e_min(S_rows):
    """E[min] from stacked survival rows: exact on the lattice."""
    Sv = np.prod(S_rows, axis=0)
    # P(min = x_k) = S_v(x_{k-1}) - S_v(x_k), with S_v(x_{-1}) = 1
    pmf = np.empty_like(Sv)
    pmf[0] = 1.0 - Sv[0]
    pmf[1:] = Sv[:-1] - Sv[1:]
    return float(pmf @ GRID)


def psi(subset, S, s, costs):
    idx = list(subset)
    return s * e_min(S[idx]) + float(costs[idx].sum())


def all_subset_psi(S, s, costs):
    """Psi for every nonempty subset at once, by bit dynamic programming:
    the survival product of mask m extends that of m without its lowest bit."""
    n_masks = 1 << N
    prods = np.empty((n_masks, len(GRID)))
    prods[0] = 1.0
    csum = np.zeros(n_masks)
    for m in range(1, n_masks):
        low = (m & -m).bit_length() - 1
        prods[m] = prods[m ^ (1 << low)] * S[low]
        csum[m] = csum[m ^ (1 << low)] + costs[low]
    pmf = np.empty_like(prods)
    pmf[:, 0] = 1.0 - prods[:, 0]
    pmf[:, 1:] = prods[:, :-1] - prods[:, 1:]
    emins = pmf @ GRID
    vals = s * emins + csum
    vals[0] = np.inf
    return vals


def exact_optimum(S, s, costs):
    vals = all_subset_psi(S, s, costs)
    m = int(np.argmin(vals))
    return float(vals[m]), {i for i in range(N) if m >> i & 1}


def greedy(S, s, costs, ground=None):
    ground = list(range(N)) if ground is None else list(ground)
    V, current = set(), np.inf
    while True:
        gains = []
        for i in ground:
            if i in V:
                continue
            gains.append((psi(V | {i}, S, s, costs), i))
        if not gains:
            break
        val, i = min(gains)
        if val >= current:
            break
        V.add(i)
        current = val
    return current, V


def local_search(S, s, costs, V0):
    """Add / drop / swap passes until no single move improves."""
    V = set(V0)
    current = psi(V, S, s, costs)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if i not in V:
                v = psi(V | {i}, S, s, costs)
                if v < current - 1e-12:
                    V.add(i); current = v; improved = True
        for i in list(V):
            if len(V) > 1:
                v = psi(V - {i}, S, s, costs)
                if v < current - 1e-12:
                    V.discard(i); current = v; improved = True
        for i in list(V):
            for j in range(N):
                if j in V or len(V) < 1:
                    continue
                v = psi((V - {i}) | {j}, S, s, costs)
                if v < current - 1e-12:
                    V.discard(i); V.add(j); current = v; improved = True
    return current, V


def top_k_by_share(shares, S, s, costs):
    """Incumbent policy, flattered: k chosen with knowledge of true Psi."""
    order = np.argsort(-shares)
    best, best_v = np.inf, None
    for k in range(1, N + 1):
        v = psi(order[:k], S, s, costs)
        if v < best:
            best, best_v = v, set(order[:k].tolist())
    return best, best_v


def best_random_k(S, s, costs, rng, draws=400):
    best = np.inf
    for k in range(1, N + 1):
        vals = []
        for _ in range(draws):
            subset = rng.choice(N, size=k, replace=False)
            vals.append(psi(subset, S, s, costs))
        best = min(best, float(np.mean(vals)))
    return best


def submodularity_check(S, s, costs, rng, trials=500):
    """phi(V, i) >= phi(W, i) for V subset W, i outside W (diminishing returns)."""
    worst = 0.0
    for _ in range(trials):
        w_size = rng.integers(2, N)
        W = set(rng.choice(N, size=w_size, replace=False).tolist())
        i = rng.choice([j for j in range(N) if j not in W])
        v_size = rng.integers(1, len(W) + 1)
        V = set(rng.choice(sorted(W), size=v_size, replace=False).tolist())
        gain = lambda A: psi(A, S, s, costs * 0.0) - psi(A | {i}, S, s, costs * 0.0)
        worst = min(worst, gain(V) - gain(W))
    return worst


def calibrated_model(shares):
    """Practice pipeline: shares + assumed Gaussian base -> implied densities."""
    base = Density(LAT, normal_pmf(0.0, 1.0))
    cal = AbilityCalibrator(base)
    offsets = np.asarray(cal.solve_from_prices([float(p) for p in shares]), dtype=float)
    pmfs = np.array([normal_pmf(a, 1.0) for a in offsets])
    return survival(pmfs)


def one_world(rng, verbose=False):
    pmfs, labels = dealer_world(rng)
    S = survival(pmfs)
    densities = [Density(LAT, p) for p in pmfs]
    shares = np.asarray(Race(densities).state_prices(), dtype=float)

    if verbose:
        for subset in [(0, 5, 10), (1, 2, 3, 4), tuple(range(N))]:
            ours = e_min(S[list(subset)])
            ref = Race([densities[i] for i in subset]).winner_density().mean()
            assert abs(ours - ref) < 1e-9, (subset, ours, ref)
        print("dealer world (E[m_i], share = P(best | all called)):")
        for i, lab in enumerate(labels):
            print(f"  {lab:14s} E[m]={float(pmfs[i] @ GRID):+.2f}  share={shares[i]:.3f}")
        worst = submodularity_check(S, 1.0, np.zeros(N), rng)
        print(f"submodularity: worst diminishing-returns violation = {worst:.2e}")

    S_cal = calibrated_model(shares)
    cost_regimes = {
        "uniform": np.full(N, 0.10),
        "heterogeneous": rng.uniform(0.04, 0.22, size=N),
    }
    rows = []
    for regime, costs in cost_regimes.items():
        for s in (2.0, 5.0, 10.0):
            ex, Vex = exact_optimum(S, s, costs)
            g, _ = greedy(S, s, costs)
            _, Vg = greedy(S, s, costs)
            ls, Vls = local_search(S, s, costs, Vg)
            _, Vc = greedy(S_cal, s, costs)
            _, Vc = local_search(S_cal, s, costs, Vc)
            cal_true = psi(Vc, S, s, costs)
            tk, Vtk = top_k_by_share(shares, S, s, costs)
            al = psi(range(N), S, s, costs)
            single = min(psi([i], S, s, costs) for i in range(N))
            scale = single - ex  # achievable savings over best single dealer
            opp_ex = sum(1 for i in Vex if labels[i].startswith("opp"))
            opp_tk = sum(1 for i in Vtk if labels[i].startswith("opp"))
            rows.append(dict(regime=regime, s=s, exact=ex, greedy=g, ls=ls,
                             cal=cal_true, topk=tk, allv=al, scale=scale,
                             opp_ex=opp_ex, opp_tk=opp_tk))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    for w in range(12):
        all_rows.extend(one_world(rng, verbose=(w == 0)))

    print(f"\n12 worlds x 2 cost regimes x 3 trade sizes; gaps to the exact")
    print(f"optimum as % of achievable savings (best-single-dealer minus exact):")
    print(f"{'regime':14s} {'s':>3} {'greedy':>8} {'grd+ls':>8} {'cal+ls':>8} "
          f"{'top-k*':>8} {'call-all':>9}  opp: V* vs top-k")
    for regime in ("uniform", "heterogeneous"):
        for s in (2.0, 5.0, 10.0):
            rs = [r for r in all_rows if r["regime"] == regime and r["s"] == s]
            gap = lambda key: 100 * np.mean([(r[key] - r["exact"]) / r["scale"] for r in rs])
            mgap = lambda key: 100 * max((r[key] - r["exact"]) / r["scale"] for r in rs)
            print(f"{regime:14s} {s:3.0f} "
                  f"{gap('greedy'):7.2f}% {gap('ls'):7.2f}% {gap('cal'):7.2f}% "
                  f"{gap('topk'):7.2f}% {gap('allv'):8.2f}%  "
                  f"{np.mean([r['opp_ex'] for r in rs]):.1f} vs "
                  f"{np.mean([r['opp_tk'] for r in rs]):.1f}   "
                  f"(worst top-k {mgap('topk'):.1f}%)")


if __name__ == "__main__":
    main()
