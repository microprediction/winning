"""Experiment 36 addendum: the free-scale factorial over the same 20 seeds.

The single-seed free-scale check (exp14 run_factorial_freescale.py) lets
each factor candidate re-optimise one scalar s on its oracle loadings,
with s chosen on a held-out half of the single deletions. This repeats
that protocol over the 20 replication seeds of experiment 36 (same truth
family, seeds 3300 + s), so the free-scale conclusion carries the same
evidential weight as the fixed-scale factorial.

Per seed and candidate: the tuning-median curve over the scale grid, the
selected s*, and held-out report medians at s = 1 and s = s*.

Run:  python experiments/exp36_factorial_replication/run_freescale_replication.py
Output: results_freescale.csv (appended per seed; restart-safe)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp14_boundaries"))
from run_boundaries import calibrate_base, factor_shares_base  # noqa: E402
from run_factorial import draw_top3, deletion_shares  # noqa: E402
from raceutil import hermite_nodes  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_freescale.csv"
N, K, N_DRAWS, SEEDS = 50, 2, 20_000_000, 20
SCALES = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8]
CANDIDATES = [("factor mixed logit", "gumbel"), ("factor probit", "normal")]


def score_blocks(base, V, D, mu_c, F, W, top3, blocks, p_menu):
    out = []
    n = len(p_menu)
    for blk in blocks:
        deleted = list(blk)   # NOT a set: np.isin treats a set as one scalar
        mass = float(sum(p_menu[i] for i in blk))
        if mass <= 0:
            continue
        keep = np.array([i for i in range(n) if i not in deleted])
        q_model = np.zeros(n)
        r, _ = factor_shares_base(mu_c, V, D, F, W, base=base, keep=keep)
        q_model[keep] = r / r.sum()
        q_true = deletion_shares(top3, deleted)
        tv = 0.5 * float(np.abs(q_model - q_true).sum())
        out.append(tv / mass)
    return np.array(out)


def one_seed(seed):
    rng = np.random.default_rng(3300 + seed)
    mu_true = rng.normal(0.0, 1.0, N)
    V_true = rng.normal(0.0, 0.6 / np.sqrt(K), (N, K))
    top3 = draw_top3(mu_true, V_true, rng, N_DRAWS)
    p_menu = np.bincount(top3[:, 0], minlength=N) / len(top3)

    F2, W2 = hermite_nodes(K)
    D_unit = np.ones(N)

    singles = [(i,) for i in range(N)]
    order = np.random.default_rng(11).permutation(N)
    tune = [b for b in (singles[i] for i in order[: N // 2])
            if p_menu[b[0]] > 5e-4]
    report = [b for b in (singles[i] for i in order[N // 2:])
              if p_menu[b[0]] > 5e-4]

    rows = []
    for name, base in CANDIDATES:
        best = None
        for s in SCALES:
            V = s * V_true
            mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
            sc = score_blocks(base, V, D_unit, mu_c, F2, W2, top3, tune, p_menu)
            med = float(np.median(sc))
            if best is None or med < best[1]:
                best = (s, med)
        s_star = best[0]
        V = s_star * V_true
        mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
        rep_star = float(np.median(score_blocks(
            base, V, D_unit, mu_c, F2, W2, top3, report, p_menu)))
        mu_1 = calibrate_base(p_menu, V_true, D_unit, F2, W2, base=base)
        rep_1 = float(np.median(score_blocks(
            base, V_true, D_unit, mu_1, F2, W2, top3, report, p_menu)))
        rows.append((seed, name, s_star, rep_1, rep_star))
    return rows


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines()[1:]:
            done.add(int(line.split(",")[0]))
    else:
        OUT.write_text("seed,model,s_star,report_median_s1,report_median_sstar\n")

    for seed in range(SEEDS):
        if seed in done:
            continue
        t0 = time.perf_counter()
        rows = one_seed(seed)
        with OUT.open("a") as f:
            for seed_, name, s_star, r1, rs in rows:
                f.write(f"{seed_},{name},{s_star},{r1:.6f},{rs:.6f}\n")
        msg = "; ".join(f"{name}: s*={s_star} s1={r1:.4f} s*={rs:.4f}"
                        for _, name, s_star, r1, rs in rows)
        print(f"seed {seed} ({time.perf_counter()-t0:.0f}s)  {msg}", flush=True)

    # summary
    import csv
    per = {}
    with OUT.open() as f:
        for row in csv.DictReader(f):
            per.setdefault(row["model"], []).append(
                (float(row["report_median_s1"]),
                 float(row["report_median_sstar"]), float(row["s_star"])))
    print("\nmodel                     med(s1)  med(s*)  s* range")
    for name, vals in per.items():
        a = np.array(vals)
        print(f"{name:24s}  {np.median(a[:,0]):.4f}   {np.median(a[:,1]):.4f}"
              f"   [{a[:,2].min():.2f}, {a[:,2].max():.2f}]")
    if len(per) == 2:
        g = np.array([v for v in per["factor mixed logit"]])
        p = np.array([v for v in per["factor probit"]])
        wins = int((p[:, 1] < g[:, 1]).sum())
        print(f"\nprobit better at free scale in {wins}/{len(p)} seeds; "
              f"median ratio {np.median(g[:,1]/p[:,1]):.2f}x")


if __name__ == "__main__":
    main()
