"""The micromanager's problem: investment, participation and exposure.

Reframing (Peter, 2026-09-24): the designer is a "micromanager" who
pays a number of forecasters and consumes their predictions. Total
effort is not the objective. The objective is the quality of the pool
of forecasts actually available, and a sole winner is a concentration
risk: if that player leaves, the platform is exposed. So the designer
values the pool's EXTREMAL quality, -E[min_i X_i] over the active
field (the best forecast it can pick), and penalises EXPOSURE, the
largest deletion value: how much of that quality rests on one member.
Both are cavity objects (cavity_calculus/NOTES.md) and both are
computed here exactly on a lattice from the TRUE performance means
(ability minus effort), not the belief.

Investment in ability under handicapped prize schedules.

exp1 found that ranking on the residual against rating (lam = 1) buys
the most effort per unit purse and keeps everyone in -- but it pays
the talent terciles almost equally, so it removes the return to BEING
good. This experiment lets ability be a stock contestants can build.

Per round each active contestant chooses
  effort     e_i  (transient; cost e^2 / (2 kappa); myopic FOC
                   e = kappa * inc_i as in exp1)
  investment u_i  (permanent: a_i <- a_i - u_i, min-wins; cost
                   u^2 / (2 kappa_u))
valued by a steady-state rule: a permanent ability gain is worth its
marginal incentive every remaining round, but once the rating has
caught up a handicap lam claws back lam of it; with a handicap LAGGED
by k rounds the gain is worth its full incentive for k rounds and
(1 - lam) of it thereafter. Horizon capped at H rounds (a discount).
    u_i = kappa_u * inc_i * [ min(k, R) + (1 - lam) * max(R - k, 0) ]
with R = min(T - t - 1, H) FUTURE rounds: investment chosen in round t
lands after round t's performance, so it can only pay in rounds
t+1 .. T-1, and in the last round it is worth nothing. (The first
version used R = min(T - t, H), a phantom extra round; its output is
kept in results.json, the corrected runs append to runs.jsonl.)

The tracker runs with positive drift so it can follow abilities that
move; belief variance then floors instead of vanishing, and hope never
dies completely under any schedule.

Designer objectives reported, all summed over rounds:
  effort    = sum of e (exp1's welfare)
  improve   = sum_i (e_i + (a_i0 - a_it)), total performance improvement
  best      = -E[min_i X_i] over the active field, relative to the
              round-0 full field at zero effort: the best forecast
  robust    = the same with the most valuable member deleted: the
              pool's quality if its top contributor walks
  exposure  = sum over rounds of the ABSOLUTE largest deletion value,
              -E[min X] minus the same after deleting the most valuable
              member (total_exposure_abs). total_exposure_change is the
              same relative to its round-0 value, i.e. best - robust
              with both measured against round 0.
plus ability gain at T by talent tercile and participation.

Sandbagging is not simulated. For a linear filter the impulse
response of the rating to a one-round shock integrates to one, so a
one-time sandbag of size s costs ~inc*s today and returns ~inc*s in
total future handicap: a wash before discounting, a loss after it.
Stated in NOTES.md; the lag variant weakens even that.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp1_prize_schedules"))
from run import schedule, incentives, N, KAPPA, C_PART, POINTS, SIGMA_A  # noqa: E402
from winning.factor import rank_probabilities  # noqa: E402
from winning.ratings.tracker import AbilityTracker  # noqa: E402

T = int(os.environ.get("CD_T", 80))
SEEDS = [0, 1, 2]
KAPPA_U = 0.01        # investment response per unit of (incentive x rounds); top gains ~0.06 sd/round at most
H = 20                # valuation horizon in rounds
DRIFT = 0.05          # tracker between-round variance growth


from scipy.stats import norm  # noqa: E402


M_DEL = 3   # deletion depths reported


def _emin_and_loo(mu, x):
    logS = norm.logsf(x[None, :] - mu[:, None])            # (n, L)
    logP = logS.sum(0)
    emin = x[0] + np.trapezoid(np.exp(logP), x)             # E[min] = a + int_a P(min > x) dx
    P_loo = np.exp(logP[None, :] - logS)                    # cavity division: product without i
    emin_loo = x[0] + np.trapezoid(P_loo, x, axis=1)
    return emin, emin_loo


def pool_quality(mu):
    """best = -E[min X] for X_i ~ N(mu_i, 1) independent (min-wins, so
    higher is better), and robust[m] = the same after greedily deleting
    the m most valuable members, m = 1..M_DEL. A pool of fewer than
    m + 1 members has robust[m] = a floor (no forecast at all)."""
    x = np.linspace(mu.min() - 7, mu.max() + 7, 4001)
    emin, emin_loo = _emin_and_loo(mu, x)
    best = -emin
    robust = []
    keep = np.ones(len(mu), bool)
    cur_loo = emin_loo
    for m in range(M_DEL):
        if keep.sum() < 2:
            robust.append(FLOOR); continue
        idx = np.flatnonzero(keep)
        j = idx[np.argmax(cur_loo[idx])]                    # most valuable = largest loss when removed
        keep[j] = False
        e, cur_loo_sub = _emin_and_loo(mu[keep], x)
        robust.append(-e)
        cur_loo = np.full(len(mu), -np.inf); cur_loo[np.flatnonzero(keep)] = cur_loo_sub
    return best, robust


FLOOR = -6.0   # quality credited to an empty pool: worse than any single forecaster


def simulate(a0, w, lam, lag, seed):
    rng = np.random.default_rng(1000 + seed)
    a = a0.copy()
    ids = [f"c{i}" for i in range(N)]
    tr = AbilityTracker(drift=DRIFT, drift_exp=1.0, init_var=SIGMA_A ** 2, beta2=1.0)
    active = np.ones(N, bool)
    m_hist = []                      # public rating means per round (max-wins negated -> min-wins)
    eff_t, qual_t, part_t, best_t, robust_t, expo_abs_t = [], [], [], [], [], []
    best0, robust0 = pool_quality(a0); robust0 = np.array(robust0)
    eff_by_i = np.zeros(N); inv_by_i = np.zeros(N); paid_by_i = np.zeros(N)
    for t in range(T):
        idx = np.flatnonzero(active)
        if len(idx) < 2:
            eff_t.append(0.0); qual_t.append(0.0); part_t.append(len(idx)); m_hist.append(np.zeros(N))
            if len(idx) == 1:
                b, r = pool_quality(a[idx]); best_t.append(b - best0)
            else:
                b = FLOOR; best_t.append(FLOOR - best0); r = [FLOOR] * M_DEL
            robust_t.append(np.array(r) - robust0); expo_abs_t.append(np.array([b - x for x in r])); continue
        m_all = np.array([-tr.rating(ids[i])[0] if t > 0 else 0.0 for i in range(N)])
        v_all = np.array([tr.rating(ids[i])[1] if t > 0 else SIGMA_A ** 2 for i in range(N)])
        m_hist.append(m_all)
        m_h = m_hist[max(t - lag, 0)]          # lagged rating for the handicap
        m, v = m_all[idx], v_all[idx]
        hcap = lam * (m_h[idx] - m_h[idx].mean())
        mu_race = m - hcap
        D = 1 + v
        wa = w[:len(idx)]
        inc = incentives(mu_race, D, wa)
        e = np.clip(KAPPA * inc, 0, None)
        R = min(T - t - 1, H)                   # future rounds in which the investment can pay
        u = np.clip(KAPPA_U * inc * (min(lag, R) + (1 - lam) * max(R - lag, 0)), 0, None)
        Rk = rank_probabilities(mu_race, D=D, points=POINTS)
        net = Rk @ wa - C_PART - e ** 2 / (2 * KAPPA) - u ** 2 / (2 * KAPPA_U)
        X = a[idx] - e + rng.normal(0, 1, len(idx))
        tr.observe([ids[i] for i in idx], t=float(t), scores=X)
        order = np.argsort(X - hcap)
        pay = np.zeros(len(idx)); pay[order] = wa
        paid_by_i[idx] += pay; eff_by_i[idx] += e; inv_by_i[idx] += u
        eff_t.append(float(e.sum()))
        qual_t.append(float((e + (a0[idx] - a[idx])).sum()))
        b, r = pool_quality(a[idx] - e)
        best_t.append(float(b - best0)); robust_t.append(np.array(r) - robust0)
        expo_abs_t.append(np.array([b - x for x in r]))
        part_t.append(int(len(idx)))
        a[idx] -= u                                # investment lands after the round
        active[idx[net < 0]] = False
    terc = np.digitize(a0, np.quantile(a0, [1 / 3, 2 / 3]))   # 0 = strongest
    robust_t = np.array(robust_t)                          # (T, M_DEL)
    expo_abs_t = np.array(expo_abs_t)                      # (T, M_DEL), absolute deletion values
    return {"total_effort": float(np.sum(eff_t)), "total_improve": float(np.sum(qual_t)),
            "total_best": float(np.sum(best_t)),
            "total_robust": [float(v) for v in robust_t.sum(0)],
            "total_exposure_change": [float(np.sum(best_t) - v) for v in robust_t.sum(0)],
            "total_exposure_abs": [float(v) for v in expo_abs_t.sum(0)],
            "baseline_exposure_round0": [float(best0 - v) for v in robust0],
            "best_path": [round(x, 3) for x in best_t[::10]],
            "robust1_path": [round(float(x), 3) for x in robust_t[::10, 0]],
            "participants_final": part_t[-1],
            "ability_gain_by_tercile": [float((a0 - a)[terc == k].sum()) for k in range(3)],
            "effort_by_tercile": [float(eff_by_i[terc == k].sum()) for k in range(3)],
            "paid_by_tercile": [float(paid_by_i[terc == k].sum()) for k in range(3)],
            "effort_path": [round(x, 3) for x in eff_t[::10]],
            "participants_path": part_t[::10]}


def check_no_phantom_round(a0):
    """Regression: in a one-round contest investment can never pay, so
    nobody invests (the first version valued a phantom extra round)."""
    global T
    T_save = T; T = 1
    try:
        r = simulate(a0, schedule("wta"), 0.0, 0, 0)
    finally:
        T = T_save
    assert max(abs(g) for g in r["ability_gain_by_tercile"]) == 0.0, r["ability_gain_by_tercile"]


def main():
    """Append-only: one JSON line per config to runs.jsonl, tagged with a
    run id and git head. results.json is the frozen first version."""
    import subprocess
    t0 = time.time()
    configs = []
    for kind in ("wta", "top3", "geometric_r0.7"):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            configs.append((kind, lam, 0))
        for lag in (5, 15, 30):
            configs.append((kind, 1.0, lag))
    fields = {s: np.sort(np.random.default_rng(s).normal(0, SIGMA_A, N)) for s in SEEDS}
    check_no_phantom_round(fields[SEEDS[0]])
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    run_id = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config = {"N": N, "T": T, "kappa": KAPPA, "kappa_u": KAPPA_U, "H": H, "drift": DRIFT, "c_part": C_PART,
              "seeds": SEEDS, "horizon": "R = min(T - t - 1, H)"}
    path = os.path.join(HERE, "runs.jsonl")
    for kind, lam, lag in configs:
        w = schedule("geometric", 0.7) if kind == "geometric_r0.7" else schedule(kind)
        runs = [simulate(fields[s], w, lam, lag, s) for s in SEEDS]
        key = f"{kind}|lam{lam}|lag{lag}"
        with open(path, "a") as fh:
            fh.write(json.dumps({"run_id": run_id, "git_head": sha, "config": config, "key": key, "runs": runs}) + "\n")
        ef = [r["total_effort"] for r in runs]; q = [r["total_best"] for r in runs]; rb = np.mean([r["total_robust"] for r in runs], 0)
        ex = np.mean([r["total_exposure_abs"][0] for r in runs])
        print(f"{key:26s} effort {np.mean(ef):7.1f}±{np.std(ef):4.1f}  best {np.mean(q):6.1f}±{np.std(q):4.1f}  robust1/2/3 {rb.round(1)}  "
              f"exposure_abs {ex:6.1f}  final n {np.mean([r['participants_final'] for r in runs]):4.1f}  "
              f"ability gain by tercile {np.mean([r['ability_gain_by_tercile'] for r in runs], 0).round(2)}  "
              f"[{time.time()-t0:.0f}s]", file=sys.stderr, flush=True)
    print(f"done run_id {run_id} in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
