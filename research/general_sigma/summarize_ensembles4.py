"""Turn results_ensembles4.csv into the paper's per-ensemble table:
median/q90/worst TV over 20 seeds, both arms, plus the kernel
stratification block. Prints LaTeX rows."""
import csv
from collections import defaultdict

import numpy as np

rows = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results_ensembles4.csv")):
    rows[r["ensemble"]][r["space"]].append(
        (float(r["tv"]), float(r["linf"]), float(r["med"])))

def fmt(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

print(f"{'ensemble':28s} arm        med(TV)   q90(TV)   worst(TV)  med(linf)")
for ens in rows:
    for arm in ("raw", "projected"):
        A = np.array(rows[ens][arm])
        if not len(A):
            continue
        print(f"{ens:28s} {arm:9s} {np.median(A[:,0]):.2e}  "
              f"{np.quantile(A[:,0],0.9):.2e}  {A[:,0].max():.2e}  "
              f"{np.median(A[:,1]):.1e}   n={len(A)}")
print()
print("LaTeX (projected arm):")
for ens in rows:
    A = np.array(rows[ens]["projected"])
    if not len(A):
        continue
    print(f"{ens} & {fmt(np.median(A[:,0]))} & {fmt(np.quantile(A[:,0],0.9))} "
          f"& {fmt(A[:,0].max())} & {fmt(np.median(A[:,2]))}\\\\")
