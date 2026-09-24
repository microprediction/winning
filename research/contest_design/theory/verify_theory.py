"""Numerical checks of the analytic results in THEORY.md.

Model: X_i = mu_i - e_i + eps_i, eps iid N(0,1) (min-wins), prizes w_k
by finishing position, effort cost e^2 / (2 kappa). Luck of the k-th
finisher L_(k) = -eps of whoever finishes k-th.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp1_prize_schedules"))
from run import incentives  # noqa: E402
from winning.factor import rank_probabilities  # noqa: E402
from winning.ratings.tracker import AbilityTracker  # noqa: E402

out = {}
rng = np.random.default_rng(0)


def position_luck_engine(mu, D=None):
    """E[L_(k)] for every k from exact rank probabilities: by Stein,
    E[L_(k)] = sum_i d P(rank_i = k) / d(-mu_i)."""
    n = len(mu); D = np.ones(n) if D is None else D
    Lk = np.zeros(n); h = 1e-4
    for i in range(n):
        e = np.zeros(n); e[i] = h
        Lk += -(rank_probabilities(mu + e, D=D, points=257)[i] - rank_probabilities(mu - e, D=D, points=257)[i]) / (2 * h)
    return Lk


def position_luck_mc(mu, M=400000):
    eps = rng.normal(0, 1, (M, len(mu))); X = mu + eps
    order = np.argsort(X, axis=1)
    return -np.take_along_axis(eps, order, axis=1).mean(0)


def nash(mu, w, kappa, iters=200, damp=0.5, tol=1e-6):
    e = np.zeros(len(mu))
    for _ in range(iters):
        e_new = np.clip(kappa * incentives(mu - e, np.ones(len(mu)), w), 0, None)
        if np.max(np.abs(e_new - e)) < tol:
            return e_new, True
        e = damp * e_new + (1 - damp) * e
    return e, False


# ---- R1/R2: total incentive = w . E[L_(k)]; symmetric field = normal order statistics
N = 24
mu0 = np.zeros(N)
Lk_eng = position_luck_engine(mu0)
Lk_mc = position_luck_mc(mu0)
# expected normal order statistics (descending) by quadrature
x = np.linspace(-9, 9, 20001); pdf = norm.pdf(x); cdf = norm.cdf(x)
from scipy.special import comb
EZ = np.array([np.trapezoid(x * N * comb(N - 1, k - 1) * cdf ** (N - k) * (1 - cdf) ** (k - 1) * pdf, x) for k in range(1, N + 1)])
out["R1_symmetric_position_luck"] = {"engine": Lk_eng.round(4).tolist(), "mc": Lk_mc.round(4).tolist(),
                                    "normal_order_stats_desc": EZ.round(4).tolist(),
                                    "max_abs_diff_engine_vs_orderstats": float(np.abs(Lk_eng - EZ).max())}
print(f"R1 symmetric N=24: E[L_(k)] engine vs normal order stats, max |diff| = {np.abs(Lk_eng - EZ).max():.2e}; "
      f"k=1..4 engine {Lk_eng[:4].round(3)} orderstats {EZ[:4].round(3)}", file=sys.stderr)

# ---- R3: symmetric Nash effort e* = kappa (w . EZ) / N, verified by best response
kappa = 3.0
res = {}
for name, w in {"wta": np.eye(N)[0], "top3": np.array([.5, .3, .2] + [0] * (N - 3)),
                "geometric_0.7": 0.7 ** np.arange(N) / (0.7 ** np.arange(N)).sum()}.items():
    e, ok = nash(mu0, w, kappa)
    pred = kappa * (w @ EZ) / N
    res[name] = {"predicted_e_star": float(pred), "best_response_e": float(e.mean()), "spread": float(e.std()), "converged": bool(ok)}
    print(f"R3 symmetric Nash {name}: predicted e* {pred:.4f}, best response {e.mean():.4f} (spread {e.std():.1e})", file=sys.stderr)
out["R3_symmetric_nash"] = res

# ---- R4: N = 2 Lazear-Rosen with unequal abilities: both exert kappa w phi(d/sqrt2)/sqrt2
res = {}
for d in [0.0, 1.0, 2.0, 3.0]:
    mu = np.array([0.0, d]); w = np.array([1.0, 0.0])
    inc = incentives(mu, np.ones(2), w)
    pred = norm.pdf(d / np.sqrt(2)) / np.sqrt(2)
    res[f"gap={d}"] = {"incentive_strong": float(inc[0]), "incentive_weak": float(inc[1]), "predicted_both": float(pred)}
    print(f"R4 N=2 gap {d}: incentives {inc.round(4)}, predicted {pred:.4f}", file=sys.stderr)
out["R4_two_player"] = res

# ---- R5: heterogeneous field: pay the position where luck matters most
a = np.sort(np.random.default_rng(0).normal(0, 1, N))
Lk_a = position_luck_engine(a)
Lk_a_mc = position_luck_mc(a)
kstar = int(np.argmax(Lk_a)) + 1
# equilibrium totals under single-position prizes and under top-3, by best response
tot = {}
for name, w in {"e_1 (wta)": np.eye(N)[0], f"e_{kstar}": np.eye(N)[kstar - 1], "e_2": np.eye(N)[1], "e_3": np.eye(N)[2],
                "top3 .5/.3/.2": np.array([.5, .3, .2] + [0] * (N - 3))}.items():
    e, ok = nash(a, w, kappa)
    tot[name] = {"total_effort": float(e.sum()), "converged": bool(ok), "static_total_incentive": float(w @ Lk_a)}
    print(f"R5 field seed0 {name:14s}: static w.L = {w @ Lk_a:.4f}, equilibrium total effort {e.sum():.4f}", file=sys.stderr)
out["R5_position_luck_heterogeneous"] = {"abilities": a.round(3).tolist(), "E_luck_by_position_engine": Lk_a.round(4).tolist(),
                                         "E_luck_by_position_mc": Lk_a_mc.round(4).tolist(), "k_star_static": kstar, "equilibrium": tot}
print(f"R5 E[L_(k)] k=1..6 engine {Lk_a[:6].round(3)} mc {Lk_a_mc[:6].round(3)}; k* = {kstar}", file=sys.stderr)

# ---- R6: Szymanski-Valletti in Gaussian form: 3 players, leader at -d, two at 0.
# static: second prize raises total incentive iff E[L_(2)] > E[L_(1)]; find d*.
def excess(d):
    L = position_luck_engine(np.array([-d, 0.0, 0.0]))
    return L[1] - L[0]
dstar = brentq(excess, 0.05, 4.0)
sv = {"d_star_static": float(dstar)}
# equilibrium: total effort vs second-prize share s at several gaps
grid = {}
for d in [0.0, 1.0, 2.0, 3.0]:
    row = {}
    for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        w = np.array([1 - s, s, 0.0]); e, ok = nash(np.array([-d, 0.0, 0.0]), w, kappa)
        row[f"s={s}"] = float(e.sum())
    best_s = max(row, key=row.get); grid[f"gap={d}"] = {"total_effort_by_s": row, "best_s": best_s}
    print(f"R6 gap {d}: total effort by second-prize share {[round(v,4) for v in row.values()]} -> best {best_s}", file=sys.stderr)
sv["equilibrium"] = grid
print(f"R6 static threshold d* = {dstar:.4f}", file=sys.stderr)
out["R6_szymanski_valletti"] = sv

# ---- R7: discouragement time, 2 players, winner-take-all, participation cost c.
# after t unit-noise observations from a unit prior the belief has variance
# v_t = 1/(t+1) and the expected rating gap is shrunk to d t/(t+1). The weak
# player quits at the first t with  d t/(t+1) / sqrt(2 (1 + v_t)) > z_c, i.e.
#   (d^2 - 2 z_c^2) t^2 - 6 z_c^2 t - 4 z_c^2 > 0,
# never if d <= sqrt(2) z_c.
c = 0.005; zc = norm.ppf(1 - c)
res = {"never_quit_below_gap": float(np.sqrt(2) * zc)}
for d in [3.0, 4.0, 5.0, 6.0]:
    A = d ** 2 - 2 * zc ** 2
    pred = np.ceil((6 * zc ** 2 + np.sqrt(36 * zc ** 4 + 16 * zc ** 2 * A)) / (2 * A)) if A > 0 else np.inf
    # simulate the tracker with the exact belief dynamics (no effort, many seeds)
    ts = []
    for seed in range(200):
        r = np.random.default_rng(seed); tr = AbilityTracker(drift=0.0, init_var=1.0, beta2=1.0)
        quit_t = None
        for t in range(400):
            m = np.array([-tr.rating(f"c{i}")[0] if t > 0 else 0.0 for i in range(2)]); v = tr.rating("c1")[1] if t > 0 else 1.0
            p_weak = norm.cdf(-(m[1] - m[0]) / np.sqrt(2 * (1 + v)))
            if p_weak < c:
                quit_t = t; break
            tr.observe(["c0", "c1"], t=float(t), scores=np.array([0.0, d]) + r.normal(0, 1, 2))
        ts.append(quit_t if quit_t is not None else np.nan)
    ts = np.array(ts, float)
    res[f"gap={d}"] = {"predicted_t_star": float(pred), "sim_median": float(np.nanmedian(ts)), "sim_mean": float(np.nanmean(ts)),
                       "frac_never_quit_within_400": float(np.isnan(ts).mean())}
    print(f"R7 gap {d}: predicted t* {pred:.2f}, simulated median {np.nanmedian(ts):.1f} mean {np.nanmean(ts):.1f} (never within 400: {np.isnan(ts).mean():.2f})", file=sys.stderr)
out["R7_discouragement_time"] = res

with open(os.path.join(HERE, "verify_results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote verify_results.json", file=sys.stderr)
