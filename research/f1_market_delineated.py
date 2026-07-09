"""The market study redone with snapshot delineation.

Post-qualifying odds embed the qualifying outcome and are not a baseline for
anything that has not seen the grid. Snapshots are therefore stratified by
timestamp relative to race date: PRE (two or more days before the race,
before qualifying in any era), SAT (the day before: ambiguous, qualifying
day in the modern era), POST (race day). Tier-matched comparisons:
  PRE  : market vs results-only ratings (both predict qualifying + race)
  POST : market vs grid-aware ratings   (both know the grid)
Both raters run in a single prequential pass; the grid-aware one observes
each race's grid order as a dt=0 event before predicting.

Run:  .venv/bin/python research/f1_market_delineated.py
"""

from __future__ import annotations

import csv
import datetime
import io
import math
import os
import zipfile
from collections import defaultdict

import numpy as np

from f1_era_slab import EraSlabThurstoneRating
from f1_grid import grid_ranks_for, load_grid, rebuild_dates
from f1_market_test import EXCLUDE, GP_MAP, norm_driver

CLIP = 1e-9


def main():
    # market with snapshot timestamps
    market = defaultdict(dict)
    snap_ts = {}
    with open(os.path.expanduser("~/.cache/winning/f1_market.csv")) as f:
        for row in csv.DictReader(f):
            if row["race_key"] in EXCLUDE:
                continue
            year, slug = row["race_key"].split("_", 1)
            gp = GP_MAP.get(slug)
            nm = norm_driver(row["driver"])
            if gp is None or "any-other" in nm:
                continue
            key = (int(year), gp)
            market[key][nm] = float(row["consensus_prob"])
            snap_ts[key] = row["snapshot_ts"]

    with zipfile.ZipFile(os.path.expanduser("~/.cache/winning/f1db-csv.zip")) as zf:
        race_meta, race_date = {}, {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                key = (int(row["year"]), row["grandPrixId"])
                race_meta[row["date"]] = key
                race_date[key] = row["date"]

    def bucket(key):
        ts = datetime.datetime.fromisoformat(snap_ts[key]).date()
        rd = datetime.date.fromisoformat(race_date[key])
        dd = (rd - ts).days
        if dd >= 2:
            return "PRE"
        if dd == 1:
            return "SAT"
        return "POST"

    dates = rebuild_dates()
    grid = load_grid()

    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    n_warm = int(len(events) * 0.2)
    sys_res = EraSlabThurstoneRating()
    sys_grid = EraSlabThurstoneRating()
    rows = []
    for idx, ev in enumerate(events):
        sys_res.elapse(ev.dt)
        sys_grid.elapse(ev.dt)
        date = dates[idx]
        gmap = grid.get(date, {})
        has_grid = len([1 for nm in ev.names if nm in gmap]) >= max(2, len(ev.names) // 2)
        if has_grid:
            sys_grid.observe(ev.names, grid_ranks_for(ev, gmap), dt=0.0)
        meta = race_meta.get(date)
        mk = market.get(meta) if meta else None
        if idx >= n_warm and mk:
            p_res = np.asarray(sys_res.win_probabilities(ev.names), dtype=float)
            p_grd = np.asarray(sys_grid.win_probabilities(ev.names), dtype=float)
            matched = [i for i, nm in enumerate(ev.names) if nm in mk]
            if len(matched) >= 8:
                winner = min(range(len(ev.ranks)), key=lambda i: ev.ranks[i])
                if winner in matched:
                    w = matched.index(winner)
                    def norm(v):
                        v = np.maximum(np.asarray(v), CLIP)
                        return v / v.sum()
                    m_p = norm([mk[ev.names[i]] for i in matched])
                    r_p = norm([p_res[i] for i in matched])
                    g_p = norm([p_grd[i] for i in matched])
                    rows.append((bucket(meta), -math.log(m_p[w]),
                                 -math.log(r_p[w]), -math.log(g_p[w])))
        sys_res.observe(ev.names, ev.ranks, dt=0.0)
        sys_grid.observe(ev.names, ev.ranks, dt=0.0)

    print(f"{'bucket':>6s} {'n':>5s} {'market':>8s} {'results':>9s} {'grid-aware':>11s} "
          f"{'mkt-vs-tier (se)':>18s}")
    for b in ("PRE", "SAT", "POST"):
        sub = [r for r in rows if r[0] == b]
        if not sub:
            continue
        mk = np.array([r[1] for r in sub])
        rs = np.array([r[2] for r in sub])
        gr = np.array([r[3] for r in sub])
        tier = rs if b == "PRE" else gr
        d = tier - mk
        print(f"{b:>6s} {len(sub):5d} {mk.mean():8.4f} {rs.mean():9.4f} {gr.mean():11.4f} "
              f"{d.mean():+10.4f} ({d.std(ddof=1)/np.sqrt(len(d)):.4f})")


if __name__ == "__main__":
    main()
