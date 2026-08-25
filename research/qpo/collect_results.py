"""Assemble everything into the one CSV the experiment brief asks for.

One row per (dataset, seed, iteration, N, rank, method), with the columns
named in the brief. Sweep rows are snapshot experiments and carry iteration 0
-- the posterior they read is the one after the initial random acquisition and
before any model-driven batch. Closed-loop rows carry their real iteration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

COLUMNS = ["dataset", "seed", "iteration", "N", "rank", "method",
           "n_mc_samples", "factorization_seconds", "pom_seconds",
           "total_seconds", "peak_memory_mb", "tv_error", "spearman",
           "top10_recall", "top100_recall", "top100_jaccard",
           "qpo_efficiency", "mean_batch_tanimoto"]


def _snapshot_meta(tag: str) -> dict:
    p = HERE / "snapshots" / tag / "meta.json"
    if p.exists():
        m = json.loads(p.read_text())
        return {"dataset": m["dataset"], "seed": m["seed"]}
    return {"dataset": tag, "seed": np.nan}


def from_sweeps() -> pd.DataFrame:
    rows = []
    for p in sorted(RES.glob("*/sweep.csv")):
        d = pd.read_csv(p)
        for tag, g in d.groupby("snapshot"):
            meta = _snapshot_meta(tag)
            gg = g.copy()
            gg["dataset"] = meta["dataset"]
            gg["seed"] = meta["seed"]
            gg["iteration"] = 0
            gg["mean_batch_tanimoto"] = gg.get("mean_tanimoto")
            if "pom_seconds" not in gg:
                gg["pom_seconds"] = gg["total_seconds"] - gg.get(
                    "factorization_seconds", 0)
            rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def from_closed_loop() -> pd.DataFrame:
    files = sorted(RES.glob("closed_loop*.csv"))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.rename(columns={"acquisition_seconds": "pom_seconds",
                          "n_scored": "N",
                          "mean_tanimoto": "mean_batch_tanimoto"})
    d["total_seconds"] = d["pom_seconds"]
    d["factorization_seconds"] = np.nan
    return d


def from_full_library() -> pd.DataFrame:
    rows = []
    for p in sorted(RES.glob("full_library_*.csv")):
        d = pd.read_csv(p)
        for tag, g in d.groupby("snapshot"):
            meta = _snapshot_meta(tag)
            gg = g.copy()
            gg["dataset"] = meta["dataset"]
            gg["seed"] = meta["seed"]
            gg["iteration"] = 0
            gg["N"] = gg["N_scored"]
            gg["pom_seconds"] = gg.get("pom_seconds", gg["seconds"])
            gg["total_seconds"] = gg["seconds"]
            gg["mean_batch_tanimoto"] = gg["mean_tanimoto"]
            rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parts = [f for f in (from_sweeps(), from_full_library(), from_closed_loop())
             if len(f)]
    if not parts:
        print("nothing to collect")
        return
    df = pd.concat(parts, ignore_index=True)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    ordered = df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]
    dest = RES / "all_results.csv"
    ordered.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(ordered)} rows, {len(ordered.columns)} columns")
    print(ordered.groupby(["dataset", "method"]).size().to_string())


if __name__ == "__main__":
    main()
