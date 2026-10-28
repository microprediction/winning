"""Iterative contest design on the ability axis.

Field of N contestants with true abilities a_i (min-wins: lower is
better). Each round, contestant i chooses effort e_i, performs
X_i = a_i - e_i + eps_i with eps ~ N(0, 1), and is paid the prize
attached to their finishing position, w_k, sum_k w_k = 1 (unit purse).
Effort costs e^2 / (2 kappa). Everyone acts on the PUBLIC belief, an
AbilityTracker fed the results; the predictive race has means m and
per-runner variance 1 + v. Handicap dial lam: positions are decided on
X_i - lam (m_i - mean m), so lam = 1 ranks on surprise against rating.

Marginal incentive (the Lazear-Rosen first-order condition for N
heterogeneous contestants and a full prize vector):
    inc_i = d E[prize_i] / d effort_i = -d (R[i] @ w) / d mu_i
computed by central differences of exact rank probabilities. The
myopic first-order condition gives e_i = kappa * inc_i. A contestant
quits for good when expected prize - participation cost - effort cost
< 0 under the public belief.

Foil: the one-shot complete-information Nash equilibrium of the same
field (true abilities known, no learning), by damped best-response
iteration on the same incentive, with and without an entry decision.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time

import numpy as np

from winning.factor import rank_probabilities
from winning.ratings.tracker import AbilityTracker

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("CD_N", 24))
T = int(os.environ.get("CD_T", 50))
SEEDS = [0, 1]
SIGMA_A = 1.0          # talent spread in units of per-round noise sd
KAPPA = 3.0            # effort e = KAPPA * marginal incentive
C_PART = 0.005         # per-round participation cost, fraction of purse
POINTS = 257
H = 1e-4


def schedule(kind, r=None):
    k = np.arange(1, N + 1)
    if kind == "wta":
        w = (k == 1).astype(float)
    elif kind == "top3":
        w = np.zeros(N); w[:3] = [0.5, 0.3, 0.2]
    elif kind == "geometric":
        w = r ** (k - 1)
    elif kind == "linear":
        w = (N - k + 1).astype(float)
    elif kind == "flat":
        w = np.ones(N)
    else:
        raise ValueError(kind)
    return w / w.sum()


def incentives(mu, D, w):
    """-d(E[prize_i])/d mu_i for each i, exact rank probabilities + FD."""
    n = len(mu); inc = np.empty(n)
    for i in range(n):
        e = np.zeros(n); e[i] = H
        up = rank_probabilities(mu + e, D=D, points=POINTS)[i] @ w
        dn = rank_probabilities(mu - e, D=D, points=POINTS)[i] @ w
        inc[i] = -(up - dn) / (2 * H)
    return inc


def one_shot_nash(a, w, entry, iters=60, damp=0.5):
    """Complete-information one-shot equilibrium by damped best response."""
    active = np.ones(N, bool)
    for _ in range(6 if entry else 1):
        idx = np.flatnonzero(active); wa = w[:len(idx)]
        e = np.zeros(len(idx))
        for _ in range(iters):
            inc = incentives(a[idx] - e, np.ones(len(idx)), wa)
            e_new = np.clip(KAPPA * inc, 0, None)
            if np.max(np.abs(e_new - e)) < 1e-4:
                e = e_new; break
            e = damp * e_new + (1 - damp) * e
        R = rank_probabilities(a[idx] - e, D=np.ones(len(idx)), points=POINTS)
        net = R @ wa - C_PART - e ** 2 / (2 * KAPPA)
        if not entry or (net >= 0).all():
            break
        active[idx[net < 0]] = False
    terc = np.digitize(a[idx], np.quantile(a, [1 / 3, 2 / 3]))
    return {"participants": int(len(idx)), "total_effort": float(e.sum()),
            "effort_by_talent_tercile": [float(e[terc == t].sum()) for t in range(3)],
            "min_net": float(net.min())}


def simulate(a, w, lam, seed):
    rng = np.random.default_rng(1000 + seed)
    ids = [f"c{i}" for i in range(N)]
    tr = AbilityTracker(drift=0.0, init_var=SIGMA_A ** 2, beta2=1.0)
    active = np.ones(N, bool)
    eff_t, part_t, spend_t = [], [], []
    eff_by_i = np.zeros(N); paid_by_i = np.zeros(N)
    for t in range(T):
        idx = np.flatnonzero(active)
        if len(idx) < 2:
            eff_t.append(0.0); part_t.append(len(idx)); spend_t.append(0.0); continue
        # AbilityTracker ratings are MAX-wins (higher is better); the race
        # and the abilities here are min-wins, so negate the mean.
        m = np.array([-tr.rating(ids[i])[0] if t > 0 else 0.0 for i in idx])
        v = np.array([tr.rating(ids[i])[1] if t > 0 else SIGMA_A ** 2 for i in idx])
        mu_race = (1 - lam) * m + lam * m.mean()
        D = 1 + v
        wa = w[:len(idx)]
        inc = incentives(mu_race, D, wa)
        e = np.clip(KAPPA * inc, 0, None)
        R = rank_probabilities(mu_race, D=D, points=POINTS)
        net = R @ wa - C_PART - e ** 2 / (2 * KAPPA)
        X = a[idx] - e + rng.normal(0, 1, len(idx))
        tr.observe([ids[i] for i in idx], t=float(t), scores=X)
        Xh = X - lam * (m - m.mean())
        order = np.argsort(Xh)
        pay = np.zeros(len(idx)); pay[order] = wa
        paid_by_i[idx] += pay; eff_by_i[idx] += e
        eff_t.append(float(e.sum())); part_t.append(int(len(idx))); spend_t.append(float(wa.sum()))
        active[idx[net < 0]] = False
    terc = np.digitize(a, np.quantile(a, [1 / 3, 2 / 3]))
    return {"total_effort": float(np.sum(eff_t)), "total_spend": float(np.sum(spend_t)),
            "effort_per_unit_spend": float(np.sum(eff_t) / max(np.sum(spend_t), 1e-12)),
            "participants_final": part_t[-1], "participants_path": part_t[::5],
            "effort_path": [round(x, 3) for x in eff_t[::5]],
            "effort_by_talent_tercile": [float(eff_by_i[terc == t].sum()) for t in range(3)],  # tercile 0 = strongest (lowest a)
            "paid_by_talent_tercile": [float(paid_by_i[terc == t].sum()) for t in range(3)]}


def main():
    t0 = time.time()
    grid = [("wta", None), ("top3", None), ("geometric", 0.5), ("geometric", 0.7),
            ("geometric", 0.85), ("geometric", 0.95), ("linear", None), ("flat", None)]
    lams = [0.0, 0.5, 1.0]
    results = {"config": {"N": N, "T": T, "sigma_a": SIGMA_A, "kappa": KAPPA,
                          "c_part": C_PART, "seeds": SEEDS}, "one_shot": {}, "iterative": {}}
    fields = {s: np.sort(np.random.default_rng(s).normal(0, SIGMA_A, N)) for s in SEEDS}
    for kind, r in grid:
        w = schedule(kind, r); key = kind if r is None else f"{kind}_r{r}"
        results["one_shot"][key] = {
            "forced_entry": [one_shot_nash(fields[s], w, entry=False) for s in SEEDS],
            "free_entry": [one_shot_nash(fields[s], w, entry=True) for s in SEEDS]}
        print(f"one-shot {key}: forced effort {np.mean([x['total_effort'] for x in results['one_shot'][key]['forced_entry']]):.3f}, "
              f"free-entry participants {np.mean([x['participants'] for x in results['one_shot'][key]['free_entry']]):.1f} "
              f"[{time.time()-t0:.0f}s]", file=sys.stderr, flush=True)
    for kind, r in grid:
        w = schedule(kind, r); key = kind if r is None else f"{kind}_r{r}"
        for lam in lams:
            runs = [simulate(fields[s], w, lam, s) for s in SEEDS]
            results["iterative"][f"{key}|lam{lam}"] = runs
            print(f"{key:16s} lam={lam:.1f}: effort {np.mean([x['total_effort'] for x in runs]):7.3f}  "
                  f"spend {np.mean([x['total_spend'] for x in runs]):6.2f}  "
                  f"final participants {np.mean([x['participants_final'] for x in runs]):5.1f}  "
                  f"[{time.time()-t0:.0f}s]", file=sys.stderr, flush=True)
            with open(os.path.join(HERE, "results.json"), "w") as fh:
                json.dump(results, fh, indent=1)
    results["runtime_s"] = time.time() - t0
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
