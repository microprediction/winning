"""Numerical checks of the analytic results in THEORY.md.

Appends one JSON line per run to verify_runs.jsonl (append-only);
verify_results.json is the frozen output of the first version (8bf6b7a).

Model: X_i = mu_i - e_i + eps_i, eps iid N(0,1) (min-wins), prizes w_k
by finishing position, effort cost e^2 / (2 kappa). Luck of the k-th
finisher L_(k) = -eps of whoever finishes k-th.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import subprocess
import sys
import time

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


def check_schedule(w):
    """The model's prize schedules: w_1 >= ... >= w_N >= 0, unit purse."""
    w = np.asarray(w, float)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12, w
    assert np.all(np.diff(w) <= 1e-15), f"non-monotone prize vector {w}"
    return w


def top_k(n, k):
    """Equal split over the top k: the vertices of the ordered unit-purse simplex."""
    w = np.zeros(n); w[:k] = 1.0 / k
    return w


def nash(mu, w, kappa, iters=200, damp=0.5, tol=1e-6, foil=False):
    """Damped best response. foil=True admits a prize vector outside the
    model (non-monotone), reported only as a labelled foil."""
    if not foil:
        check_schedule(w)
    e = np.zeros(len(mu))
    for _ in range(iters):
        e_new = np.clip(kappa * incentives(mu - e, np.ones(len(mu)), w), 0, None)
        if np.max(np.abs(e_new - e)) < tol:
            return e_new, True
        e = damp * e_new + (1 - damp) * e
    return e, False


# ---- Numerics: the engine is a finite lattice (points) and the incentive a
# central difference (H). Neither is exact; check convergence in both.
import run as _run  # noqa: E402
conv = []
a_conv = np.sort(np.random.default_rng(0).normal(0, 1, 24))
w_top3 = np.array([.5, .3, .2] + [0] * 21); w_wta = np.eye(24)[0]
for P in (129, 257, 513, 1025):
    for HH in (1e-3, 1e-4, 1e-5):
        _run.POINTS, _run.H = P, HH
        i_h = incentives(a_conv, np.ones(24), w_top3); i_s = incentives(np.zeros(24), np.ones(24), w_wta)
        conv.append({"points": P, "H": HH, "hetero_top3_total": float(i_h.sum()), "hetero_top3_favourite": float(i_h[0]),
                     "symmetric_wta_total": float(i_s.sum())})
_run.POINTS, _run.H = 257, 1e-4
ref = [c for c in conv if c["points"] == 1025 and c["H"] == 1e-5][0]
spread = max(abs(c[k] - ref[k]) for c in conv if c["H"] <= 1e-4 for k in ("hetero_top3_total", "hetero_top3_favourite", "symmetric_wta_total"))
print(f"numerics: max |incentive(points, H) - incentive(1025, 1e-5)| over points 129..1025, H <= 1e-4: {spread:.1e}", file=sys.stderr)
# location-scale: symmetric-field incentives scale as 1/sqrt(D), D = 1 + v
scal = {}
for Dv in (1.0, 2.0, 4.0):
    sdv = float(incentives(np.zeros(24), Dv * np.ones(24), w_wta).sum())
    scal[f"D={Dv}"] = {"total_incentive": sdv, "times_sqrt_D": sdv * np.sqrt(Dv)}
    print(f"numerics: symmetric N=24 wta, D = {Dv}: total incentive {sdv:.8f}, x sqrt(D) = {sdv * np.sqrt(Dv):.8f}", file=sys.stderr)
out["numerics_convergence"] = {"grid": conv, "max_abs_diff_vs_1025_1e-5_for_H_le_1e-4": spread}
out["numerics_scale_1_over_sqrt_D"] = scal

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
    res[f"gap={d}"] = {"incentive_strong": float(inc[0]), "incentive_weak": float(inc[1]), "predicted_incentive_both": float(pred),
                       "effort_each_kappa3": float(kappa * pred), "total_effort_kappa3": float(2 * kappa * pred)}
    print(f"R4 N=2 gap {d}: incentives {inc.round(4)}, predicted {pred:.4f}; at kappa={kappa:g} effort each {kappa * pred:.4f}, "
          f"total {2 * kappa * pred:.5f}", file=sys.stderr)
out["R4_two_player"] = res

# ---- R5: heterogeneous field: pay the position where luck matters most
a = np.sort(np.random.default_rng(0).normal(0, 1, N))
Lk_a = position_luck_engine(a)
Lk_a_mc = position_luck_mc(a)
kstar = int(np.argmax(Lk_a)) + 1                 # unconstrained vertex: NOT a feasible schedule unless kstar = 1
prefix_avg = np.cumsum(Lk_a) / np.arange(1, N + 1)   # static total incentive of each feasible vertex top_k
k_static = int(np.argmax(prefix_avg)) + 1
print(f"R5 static: max prefix average of E[L_(k)] at k = {k_static} ({prefix_avg[k_static - 1]:.4f}); "
      f"prefix averages k=1..6 {prefix_avg[:6].round(4)}", file=sys.stderr)
# equilibrium totals over the feasible vertices (equal top-k splits), top-3 .5/.3/.2,
# and, as labelled non-monotone foils outside the model, single-position prizes.
tot = {}
K_EQ = 8
for k in range(1, K_EQ + 1):
    w = top_k(N, k); e, ok = nash(a, w, kappa)
    tot[f"top{k} equal"] = {"total_effort": float(e.sum()), "converged": bool(ok), "static_total_incentive": float(w @ Lk_a), "feasible": True}
    print(f"R5 field seed0 top{k} equal     : static w.L = {w @ Lk_a:.4f}, equilibrium total effort {e.sum():.4f}", file=sys.stderr)
w = check_schedule(np.array([.5, .3, .2] + [0] * (N - 3))); e, ok = nash(a, w, kappa)
tot["top3 .5/.3/.2"] = {"total_effort": float(e.sum()), "converged": bool(ok), "static_total_incentive": float(w @ Lk_a), "feasible": True}
print(f"R5 field seed0 top3 .5/.3/.2   : static w.L = {w @ Lk_a:.4f}, equilibrium total effort {e.sum():.4f}", file=sys.stderr)
for k in (2, 3):
    w = np.eye(N)[k - 1]; e, ok = nash(a, w, kappa, foil=True)
    tot[f"FOIL pay position {k} only (non-monotone, outside the model)"] = {
        "total_effort": float(e.sum()), "converged": bool(ok), "static_total_incentive": float(w @ Lk_a), "feasible": False}
    print(f"R5 field seed0 FOIL e_{k} only   : static w.L = {w @ Lk_a:.4f}, equilibrium total effort {e.sum():.4f}", file=sys.stderr)
feas = {kk: vv for kk, vv in tot.items() if vv["feasible"]}
best_feas = max(feas, key=lambda kk: feas[kk]["total_effort"])
print(f"R5 best feasible schedule in equilibrium: {best_feas} ({feas[best_feas]['total_effort']:.4f})", file=sys.stderr)
out["R5_position_luck_heterogeneous"] = {"abilities": a.round(3).tolist(), "E_luck_by_position_engine": Lk_a.round(4).tolist(),
                                         "E_luck_by_position_mc": Lk_a_mc.round(4).tolist(),
                                         "argmax_k_unconstrained": kstar, "prefix_average_static": prefix_avg.round(4).tolist(),
                                         "k_static_feasible": k_static, "equilibrium": tot, "best_feasible_equilibrium": best_feas}
print(f"R5 E[L_(k)] k=1..6 engine {Lk_a[:6].round(3)} mc {Lk_a_mc[:6].round(3)}", file=sys.stderr)

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
        w = check_schedule(np.array([1 - s, s, 0.0])); e, ok = nash(np.array([-d, 0.0, 0.0]), w, kappa)
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

# Append-only record: one JSON line per run. verify_results.json is the
# frozen output of the first version (8bf6b7a) and is never rewritten.
try:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", HERE], cwd=HERE, capture_output=True, text=True).stdout.strip())
except OSError:
    sha, dirty = "", None
rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": sha, "worktree_dirty": dirty, "results": out}
with open(os.path.join(HERE, "verify_runs.jsonl"), "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("appended to verify_runs.jsonl", file=sys.stderr)
