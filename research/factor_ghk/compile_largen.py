"""Compile runs/largen_n*.csv into markdown tables for RESULTS.md part 2.

    python research/factor_ghk/compile_largen.py
"""
import csv, glob, os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(HERE, "runs", "largen_n*.csv")), key=lambda p: int(os.path.basename(p)[8:-4])):
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    r0 = rows[0]
    n = int(r0["n"])
    # keep the LAST occurrence of each (method, target) (a resumed run appends again)
    last = OrderedDict()
    for r in rows:
        last[(r["method"], r["target"])] = r
    methods = list(OrderedDict.fromkeys(m for m, _ in last))
    targets = list(OrderedDict.fromkeys(t for _, t in last))
    truth = {t: float(last[(methods[0], t)]["truth_p"]) for t in targets}
    targets.sort(key=lambda t: -truth[t])
    print(f"\n### n = {n:,}, rank {r0['rank']}, D = {r0['D']}, sharpness {float(r0['sharp']):.1f}\n")
    print(f"Truth: Sobol 2^{r0['truth_m']} (seed 101), vs second scramble {float(r0['truth_vs_scramble']):.1e}, vs MC {float(r0['truth_vs_mc']):.1e}. "
          f"Read nothing below {2*float(r0['truth_vs_scramble']):.0e}.\n")
    print("| method | passes | per target | max abs err | max rel err | wall s |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        rs = [last[(m, t)] for t in targets if (m, t) in last]
        passes = int(rs[0]["passes"]); budget = int(rs[0]["budget"])
        per = budget if "Hybrid" in m else ("-" if "ghk" in m else passes)
        print(f"| {m} | {passes if passes else '-'} | {per} | {max(float(r['abs_err']) for r in rs):.1e} | {max(float(r['rel_err']) for r in rs):.1e} | {float(rs[0]['seconds']):.1f} |")
    print("\nPer-target relative error, favourites left, longshots right:\n")
    print("| method | " + " | ".join(f"{truth[t]:.1e}" for t in targets) + " |")
    print("|---|" + "---:|" * len(targets))
    for m in methods:
        if not any((m, t) in last for t in targets):
            continue
        print(f"| {m} | " + " | ".join(f"{float(last[(m, t)]['rel_err']):.1e}" if (m, t) in last else "-" for t in targets) + " |")
