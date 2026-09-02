"""select_race_group: the group-entry optimizer, prototyped and refereed.

Semantics under test (min-wins): choose a group S of entrants -- the
unselected candidates DO NOT run -- to maximize the probability that
the group produces the winner against an outside competitor,

    f(S) = P(min_{i in S} X_i < X_out)
         = 1 - E_z int f_out(t|z) prod_{i in S} S_i(t|z) dt,

subject to sum costs <= budget and |S| <= max_size.

Structure exploited:
  * f is monotone submodular: conditional on (z, t) the group-miss
    probability is a product of per-member terms in [0,1], and
    Delta_i(S) = E[ f_out * prod_S S_j * (1 - S_i) ] is pointwise
    dominated as S grows. (Checked numerically below as well.)
  * One field pass prices ALL n marginal gains: W(t|z) =
    f_out(t|z) prod_{j in S} S_j(t|z) once, then Delta_i =
    int W (1 - S_i) for every i simultaneously -- the cavity trick
    applied to subset selection. Greedy is O(k n L Q) total.

Methods refereed:
  greedy       cost-ratio greedy + best-feasible-singleton safeguard
  prefix       sort by mu ascending, best feasible prefix
  brute force  exact enumeration at n <= 14 (the referee)

Checks: submodularity and monotonicity on random nested pairs; f(S)
against Monte Carlo; greedy vs exact optimum (approximation ratio);
the two-cluster diversification demo (greedy spreads across clusters,
prefix piles into one); and an n=5000 timing.
"""
import itertools
import json
import os
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

from scipy.stats import norm  # noqa: E402

QZ = 21
L = 1401


def gh_nodes(q):
    x, w = np.polynomial.hermite_e.hermegauss(q)
    return x, w / w.sum()


class GroupRace:
    """Precomputes per-node fields; prices f(S) and all gains fast."""

    def __init__(self, mu, V, d, mu_out, v_out, sd_out):
        self.n = len(mu)
        assert V.ndim == 2 and V.shape[1] == 1, "prototype is rank-1"
        z, wq = gh_nodes(QZ)
        sd = np.sqrt(d)
        node_means = mu[None, :] + np.outer(z, V[:, 0])
        node_out = mu_out + z * v_out.sum()
        lo = min(node_means.min(), node_out.min()) \
            - 8 * max(sd.max(), sd_out)
        hi = max(node_means.max(), node_out.max()) \
            + 8 * max(sd.max(), sd_out)
        self.x = np.linspace(lo, hi, L)
        self.h = self.x[1] - self.x[0]
        self.wq = wq
        # per node: logS (n, L) for candidates, f_out (L,)
        self.logS = []
        self.fout = []
        for zi in z:
            m = mu + V[:, 0] * zi
            t = (self.x[None, :] - m[:, None]) / sd[:, None]
            self.logS.append(norm.logsf(t))
            mo = mu_out + v_out.sum() * zi
            self.fout.append(np.exp(norm.logpdf(
                (self.x - mo) / sd_out)) / sd_out)

    def value(self, S):
        S = list(S)
        miss = 0.0
        for wq, logS, fout in zip(self.wq, self.logS, self.fout):
            miss += wq * self.h * float(
                fout @ np.exp(logS[S].sum(0)))
        return 1.0 - miss

    def gains(self, S):
        """Delta_i(S) for every candidate i, one pass per node."""
        S = list(S)
        g = np.zeros(self.n)
        for wq, logS, fout in zip(self.wq, self.logS, self.fout):
            W = fout * np.exp(logS[S].sum(0)) if S else fout
            g += wq * self.h * (W @ (1.0 - np.exp(logS)).T)
        return g


def greedy(gr, costs, budget, max_size):
    S, spent = [], 0.0
    chosen = np.zeros(gr.n, bool)
    while len(S) < max_size:
        g = gr.gains(S)
        ratio = np.where(chosen | (costs + spent > budget),
                         -np.inf, g / costs)
        i = int(ratio.argmax())
        if not np.isfinite(ratio[i]) or g[i] <= 1e-15:
            break
        S.append(i)
        chosen[i] = True
        spent += costs[i]
    # knapsack safeguard: best feasible singleton
    g0 = gr.gains([])
    feas = costs <= budget
    if feas.any():
        best_single = int(np.where(feas, g0, -np.inf).argmax())
        if gr.value([best_single]) > gr.value(S):
            S = [best_single]
    return S


def prefix(gr, mu, costs, budget, max_size):
    order = np.argsort(mu)                      # min-wins: small mu best
    S, spent = [], 0.0
    for i in order:
        if len(S) >= max_size or spent + costs[i] > budget:
            break
        S.append(int(i))
        spent += costs[i]
    return S


def brute(gr, costs, budget, max_size):
    best, best_v = [], -1.0
    idx = range(gr.n)
    for k in range(1, max_size + 1):
        for S in itertools.combinations(idx, k):
            if costs[list(S)].sum() <= budget:
                v = gr.value(S)
                if v > best_v:
                    best, best_v = list(S), v
    return best, best_v


def mc_value(mu, V, d, mu_out, v_out, sd_out, S, n_mc, seed):
    rng = np.random.default_rng(seed)
    r = V.shape[1]
    z = rng.normal(size=(n_mc, r))
    Xs = (mu[S][None, :] + z @ V[S].T
          + rng.normal(size=(n_mc, len(S))) * np.sqrt(d[S]))
    Xo = mu_out + z @ v_out + rng.normal(size=n_mc) * sd_out
    return float((Xs.min(1) < Xo).mean())


if __name__ == "__main__":
    results = {}
    rng = np.random.default_rng(5)

    # --- referee configuration: n=12, rank-1, random costs ---
    n = 12
    mu = rng.normal(0, 0.7, n)
    V = rng.normal(0, 0.6, (n, 1))
    d = 0.3 + rng.random(n)
    costs = 0.5 + rng.random(n)
    budget, max_size = 2.2, 5
    gr = GroupRace(mu, V, d, mu_out=-0.6, v_out=np.array([0.4]),
                   sd_out=0.8)

    # value vs MC on three random sets
    errs = []
    for k, seed in ((2, 1), (4, 2), (5, 3)):
        S = list(rng.choice(n, k, replace=False))
        errs.append(abs(gr.value(S) - mc_value(
            mu, V, d, -0.6, np.array([0.4]), 0.8, S, 2_000_000, seed)))
    print(f"[value vs MC] max|err| {max(errs):.2e}")

    # submodularity + monotonicity, 500 nested trials
    worst = 0.0
    for _ in range(500):
        k = rng.integers(1, 6)
        T = list(rng.choice(n, k + 2, replace=False))
        S = T[:k]
        i = int(rng.choice([j for j in range(n) if j not in T]))
        dS = gr.value(S + [i]) - gr.value(S)
        dT = gr.value(T + [i]) - gr.value(T)
        worst = max(worst, dT - dS)
        assert dS >= -1e-12, "monotonicity violated"
    print(f"[submodularity] worst Delta_i(T)-Delta_i(S) over 500 nested "
          f"trials: {worst:.2e} (<=0 required up to quadrature noise)")

    # greedy and prefix vs exact
    Sg = greedy(gr, costs, budget, max_size)
    Sp = prefix(gr, mu, costs, budget, max_size)
    Sb, vb = brute(gr, costs, budget, max_size)
    vg, vp = gr.value(Sg), gr.value(Sp)
    print(f"[n12 exact referee] greedy {vg:.5f}  prefix {vp:.5f}  "
          f"optimum {vb:.5f}  greedy ratio {vg / vb:.4f}")
    results["n12"] = dict(greedy=vg, prefix=vp, optimum=vb,
                          ratio=vg / vb, mc_max_err=float(max(errs)),
                          submodularity_worst=float(worst))

    # --- diversification demo: two tight clusters ---
    n = 40
    mu = np.r_[np.full(20, -0.5), np.full(20, -0.45)]
    V = np.r_[np.tile([[0.9]], (20, 1)), np.tile([[-0.9]], (20, 1))]
    d = np.full(n, 0.15)
    costs = np.ones(n)
    gr2 = GroupRace(mu, V, d, mu_out=-1.2, v_out=np.array([0.0]),
                    sd_out=0.6)
    Sg = greedy(gr2, costs, budget=4, max_size=4)
    Sp = prefix(gr2, mu, costs, budget=4, max_size=4)
    n_a = sum(1 for i in Sg if i < 20)
    print(f"[clusters] greedy picks {n_a} from cluster A, {4 - n_a} from "
          f"B: f={gr2.value(Sg):.5f}; prefix (all one cluster) "
          f"f={gr2.value(Sp):.5f}")
    results["clusters"] = dict(greedy=gr2.value(Sg),
                               prefix=gr2.value(Sp),
                               greedy_split=[n_a, 4 - n_a])

    # --- scale: n=5000, pick 10 ---
    n = 5000
    mu = rng.normal(0, 0.7, n)
    V = rng.normal(0, 0.6, (n, 1))
    d = 0.3 + rng.random(n)
    costs = 0.5 + rng.random(n)
    gr3 = GroupRace(mu, V, d, mu_out=-1.5, v_out=np.array([0.5]),
                    sd_out=0.8)
    t0 = time.time()
    Sg = greedy(gr3, costs, budget=8.0, max_size=10)
    dt = time.time() - t0
    print(f"[n=5000] greedy picked {len(Sg)} in {dt:.2f}s, "
          f"f={gr3.value(Sg):.5f}")
    results["n5000"] = dict(seconds=dt, size=len(Sg),
                            value=gr3.value(Sg))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
