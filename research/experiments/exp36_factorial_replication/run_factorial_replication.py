"""Experiment 36: the misspecified factorial, replicated over 20 seeds.

Experiment 14's factorial (t(5) factors, standardized skew-normal
idiosyncratic noise, misspecified for every candidate, oracle loadings,
common full-menu calibration) ran on a single truth design and seed --
a caveat the paper states. This replication repeats the identical
protocol over 20 seeds: fresh mu*, V*, and 2e7 common truth draws per
seed, all 50 single deletions plus 100 fixed pairs, four candidates.

Reported per stratum: median-across-seeds of the per-seed mean
misallocated fraction for each candidate, and the count of seeds in
which factor probit beats plain logit.

Run:  python experiments/exp36_factorial_replication/run_factorial_replication.py
Output: results.csv, summary.txt (appended per seed; restart-safe)
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
N, K, N_DRAWS, SEEDS = 50, 2, 20_000_000, 20
STRATA = [(">10%", 0.10, 10.0), ("2-10%", 0.02, 0.10),
          ("0.5-2%", 0.005, 0.02), ("0.05-0.5%", 0.0005, 0.005)]
NAMES = ["independent Luce", "independent probit",
         "factor mixed logit", "factor probit"]


def one_seed(seed):
    rng = np.random.default_rng(3300 + seed)
    mu_true = rng.normal(0.0, 1.0, N)
    V_true = rng.normal(0.0, 0.6 / np.sqrt(K), (N, K))
    top3 = draw_top3(mu_true, V_true, rng, N_DRAWS)
    p_menu = np.bincount(top3[:, 0], minlength=N) / len(top3)

    F2, W2 = hermite_nodes(2)
    D_unit = np.ones(N)
    Vz = np.zeros((N, K))
    models = {}
    for name, base, V in [("independent Luce", "gumbel", Vz),
                          ("independent probit", "normal", Vz),
                          ("factor mixed logit", "gumbel", V_true),
                          ("factor probit", "normal", V_true)]:
        mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
        models[name] = (base, V, mu_c)

    blocks = [(i,) for i in range(N)]
    prng = np.random.default_rng(4)
    seen = set()
    while len(seen) < 100:
        i, j = sorted(prng.choice(N, 2, replace=False))
        seen.add((int(i), int(j)))
    blocks += sorted(seen)

    per_obs = []
    for B in blocks:
        mass = float(p_menu[list(B)].sum())
        if mass < 5e-4:
            continue
        q_true = deletion_shares(top3, list(B))
        keep = np.setdiff1d(np.arange(N), B)
        for name, (base, V, mu_c) in models.items():
            q, _ = factor_shares_base(mu_c, V, D_unit, F2, W2,
                                      base=base, keep=keep)
            full = np.zeros(N); full[keep] = q
            tv = 0.5 * float(np.abs(full - q_true).sum())
            per_obs.append((name, mass, tv / mass))

    out = {}
    for lab, lo, hi in STRATA:
        for nm in NAMES:
            sel = [r for n_, m, r in per_obs if n_ == nm and lo < m <= hi]
            out[(nm, lab)] = float(np.mean(sel)) if sel else np.nan
    return out


def main():
    res_path = HERE / "results.csv"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines()[1:]:
            done.add(int(line.split(",")[0]))
    else:
        hdr = "seed," + ",".join(
            f"{nm.replace(' ', '_')}_{lab}" for nm in NAMES
            for lab, _, _ in STRATA)
        res_path.write_text(hdr + "\n")

    for seed in range(SEEDS):
        if seed in done:
            continue
        t0 = time.time()
        out = one_seed(seed)
        row = f"{seed}," + ",".join(
            f"{out[(nm, lab)]:.5f}" for nm in NAMES for lab, _, _ in STRATA)
        with open(res_path, "a") as f:
            f.write(row + "\n")
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)

    # aggregate
    import csv
    rows = list(csv.DictReader(open(res_path)))
    lines = []
    for lab, _, _ in STRATA:
        med = {nm: np.median([float(r[f"{nm.replace(' ', '_')}_{lab}"])
                              for r in rows]) for nm in NAMES}
        wins = sum(float(r[f"factor_probit_{lab}"])
                   < float(r[f"independent_Luce_{lab}"]) for r in rows)
        lines.append(f"{lab}: " + "  ".join(
            f"{nm}={med[nm]:.3f}" for nm in NAMES)
            + f"  [factor probit beats plain logit {wins}/{len(rows)}]")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
