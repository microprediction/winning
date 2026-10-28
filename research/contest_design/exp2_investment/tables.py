"""Print NOTES.md's exp2 tables from the latest run in runs.jsonl
(append-only; pass a run_id to pick another). Means over seeds."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "runs.jsonl"))]
run_id = sys.argv[1] if len(sys.argv) > 1 else rows[-1]["run_id"]
rows = {r["key"]: r["runs"] for r in rows if r["run_id"] == run_id}
print(f"run_id {run_id}, {len(rows)} configs")
NAME = {"wta": "wta", "top3": "top3", "geometric_r0.7": "geom .7"}


def m(runs, f):
    return np.mean([f(r) for r in runs], 0)


print("schedule | lam | lag   effort  improve   best  robust1  expo_abs  expo_chg  final n   ability gain (S / M / W)")
for key, runs in rows.items():
    kind, lam, lag = key.split("|")
    g = m(runs, lambda r: r["ability_gain_by_tercile"])
    print(f"{NAME[kind]:8s} | {lam[3:]:4s}| {lag[3:]:3s} {m(runs, lambda r: r['total_effort']):7.0f} {m(runs, lambda r: r['total_improve']):7.0f}"
          f" {m(runs, lambda r: r['total_best']):7.1f} {m(runs, lambda r: r['total_robust'][0]):7.1f}"
          f" {m(runs, lambda r: r['total_exposure_abs'][0]):8.1f} {m(runs, lambda r: r['total_exposure_change'][0]):8.1f}"
          f" {m(runs, lambda r: r['participants_final']):7.1f}    {g[0]:4.1f} {g[1]:4.1f} {g[2]:4.1f}")
print()
print("schedule | lam | lag    best  robust1 robust2 robust3  improve  final n")
for key, runs in rows.items():
    kind, lam, lag = key.split("|")
    rb = m(runs, lambda r: r["total_robust"])
    print(f"{NAME[kind]:8s} | {lam[3:]:4s}| {lag[3:]:3s} {m(runs, lambda r: r['total_best']):7.1f} {rb[0]:7.1f} {rb[1]:7.1f} {rb[2]:7.1f}"
          f" {m(runs, lambda r: r['total_improve']):7.0f} {m(runs, lambda r: r['participants_final']):7.1f}")
