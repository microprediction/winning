"""Paired comparison of the two fitting arms in results_ensembles4.csv.

For each named ensemble the per-seed quantity is the full-vector total
variation against that seed's simulation referee. The difference is raw
minus identified ("projected" in the CSV), so positive means the
identified arm is better. For each ensemble this prints the median
paired difference, a distribution-free 95% interval for the centre of
the differences (the Hodges-Lehmann interval: order statistics of the
Walsh averages, normal-approximation rank cut), the number of seeds on
which the identified arm wins, the two-sided Wilcoxon signed-rank
p-value, and its Holm adjustment over the ensembles.

Reads the committed CSV. No random numbers. Rows without a score (a
case that raised) are skipped and counted.

    python paired_ensembles4.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

tv = defaultdict(dict)
skipped = 0
for r in csv.DictReader(open("results_ensembles4.csv")):
    try:
        tv[(r["ensemble"], r["space"])][int(r["seed"])] = float(r["tv"])
    except (TypeError, ValueError):
        skipped += 1

rows = []
for ens in sorted({e for e, _ in tv}):
    a, b = tv.get((ens, "raw")), tv.get((ens, "projected"))
    if not a or not b:
        continue
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    walsh = np.sort([(d[i] + d[j]) / 2
                     for i in range(n) for j in range(i, n)])
    cut = max(int(np.floor(n * (n + 1) / 4 - 1.959964
                           * np.sqrt(n * (n + 1) * (2 * n + 1) / 24))), 0)
    lo, hi = walsh[cut], walsh[len(walsh) - 1 - cut]
    p = wilcoxon(d, alternative="two-sided").pvalue
    rows.append((ens, n, float(np.median(d)), lo, hi, int((d > 0).sum()), p))

m = len(rows)
holm = np.empty(m)
running = 0.0
for rank, idx in enumerate(np.argsort([r[-1] for r in rows])):
    running = max(running, min(1.0, (m - rank) * rows[idx][-1]))
    holm[idx] = running

print(f"{'ensemble':30s}  n  med(raw-ident)  95% HL interval          "
      f"ident wins  p 2-sided  Holm")
for (ens, n, med, lo, hi, wins, p), h in zip(rows, holm):
    print(f"{ens:30s} {n:2d}  {med:+.2e}      [{lo:+.2e}, {hi:+.2e}]  "
          f"{wins:2d}/{n}       {p:.1e}    {h:.1e}")
better = sum(1 for r, h in zip(rows, holm) if h < 0.05 and r[2] > 0)
worse = sum(1 for r, h in zip(rows, holm) if h < 0.05 and r[2] < 0)
raw05 = sum(1 for r in rows if r[-1] < 0.05 and r[2] > 0)
print(f"\nensembles {m}; rows skipped {skipped}")
print(f"identified better: {raw05} at unadjusted 0.05, {better} at Holm 0.05; "
      f"raw better at Holm 0.05: {worse}")
