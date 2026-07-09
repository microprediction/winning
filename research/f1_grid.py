"""Grid positions as information: qualifying is a contest too.

The market test's biggest confound is that many odds snapshots postdate
qualifying while the rater ignores grid position entirely. The fix needs no
new machinery: feed each race's grid order to the rater as a ranking event
(dt=0) BEFORE predicting the race. Qualifying precedes both the race and
the snapshots, so this is prequentially legitimate and puts the ratings on
the market's information tier.

Reports (a) the standalone improvement over all grands prix and (b) the
market test re-run with the grid-aware rater.

Run:  .venv/bin/python research/f1_grid.py
"""

from __future__ import annotations

import csv
import io
import math
import os
import zipfile
from collections import defaultdict

import numpy as np

from f1_era_slab import EraSlabThurstoneRating
from f1_market_test import EXCLUDE, GP_MAP, norm_driver
from winning.benchmarks.metrics import Metrics

CLIP = 1e-9


def load_grid():
    """raceId-date -> {driverId: grid position} from f1db."""
    path = os.path.expanduser("~/.cache/winning/f1db-csv.zip")
    with zipfile.ZipFile(path) as zf:
        rd = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rd[row["id"]] = row["date"]
        grid = defaultdict(dict)
        with zf.open("f1db-races-race-results.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                date = rd.get(row["raceId"])
                if not date:
                    continue
                g = row.get("gridPositionNumber")
                if g:
                    grid[date].setdefault(row["driverId"], int(g))
    return grid


def rebuild_dates():
    path = os.path.expanduser("~/.cache/winning/f1db-csv.zip")
    with zipfile.ZipFile(path) as zf:
        rd = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rd[row["id"]] = row["date"]
        by_race = defaultdict(list)
        with zf.open("f1db-races-race-results.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                by_race[row["raceId"]].append(row)
    dates = []
    for race_id, rows in by_race.items():
        date = rd.get(race_id)
        if not date:
            continue
        seen = set()
        for row in sorted(rows, key=lambda r: int(r.get("positionDisplayOrder") or 0)):
            seen.add(row["driverId"])
        if len(seen) >= 2:
            dates.append(date)
    dates.sort()
    return dates


def grid_ranks_for(ev, grid_map):
    """Competition ranks from grid positions; ungridded entrants tied last."""
    gps = [grid_map.get(nm) for nm in ev.names]
    known = sorted(g for g in gps if g is not None)
    ranks = []
    worst = len(known) + 1
    for g in gps:
        ranks.append(worst if g is None else 1 + known.index(g))
    return ranks


def run(use_grid: bool, market=None, race_meta=None, dates=None, grid=None):
    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    n_warm = int(len(events) * 0.2)
    system = EraSlabThurstoneRating()
    m = Metrics()
    scored_mkt = []
    for idx, ev in enumerate(events):
        system.elapse(ev.dt)
        date = dates[idx]
        if use_grid:
            gmap = grid.get(date, {})
            if len([1 for nm in ev.names if nm in gmap]) >= max(2, len(ev.names) // 2):
                system.observe(ev.names, grid_ranks_for(ev, gmap), dt=0.0)
        if idx >= n_warm:
            probs = np.asarray(system.win_probabilities(ev.names), dtype=float)
            holders = [i for i, r in enumerate(ev.ranks) if r == min(ev.ranks)]
            if len(holders) == 1:
                mus = [system.rating(nm).mu for nm in ev.names]
                m.score_event(list(probs), ev.ranks, mus)
            mk = market.get(race_meta.get(date)) if market else None
            if mk:
                matched, m_p, s_p = [], [], []
                for i, nm in enumerate(ev.names):
                    if nm in mk:
                        matched.append(i); m_p.append(mk[nm]); s_p.append(probs[i])
                if len(matched) >= 8:
                    winner = min(range(len(ev.ranks)), key=lambda i: ev.ranks[i])
                    if winner in matched:
                        w = matched.index(winner)
                        m_p = np.maximum(np.asarray(m_p), CLIP); m_p /= m_p.sum()
                        s_p = np.maximum(np.asarray(s_p), CLIP); s_p /= s_p.sum()
                        scored_mkt.append((-math.log(m_p[w]), -math.log(s_p[w]), m_p, s_p, w))
        system.observe(ev.names, ev.ranks, dt=0.0)
    return m.summary(), scored_mkt


def main():
    market = defaultdict(dict)
    with open(os.path.expanduser("~/.cache/winning/f1_market.csv")) as f:
        for row in csv.DictReader(f):
            if row["race_key"] in EXCLUDE:
                continue
            year, slug = row["race_key"].split("_", 1)
            gp = GP_MAP.get(slug)
            nm = norm_driver(row["driver"])
            if gp is None or "any-other" in nm:
                continue
            market[(int(year), gp)][nm] = float(row["consensus_prob"])
    path = os.path.expanduser("~/.cache/winning/f1db-csv.zip")
    with zipfile.ZipFile(path) as zf:
        race_meta = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                race_meta[row["date"]] = (int(row["year"]), row["grandPrixId"])
    dates = rebuild_dates()
    grid = load_grid()

    for label, use_grid in [("results only", False), ("results + grid", True)]:
        s, scored = run(use_grid, market, race_meta, dates, grid)
        mk = np.array([r[0] for r in scored]); md = np.array([r[1] for r in scored])
        d = md - mk
        print(f"{label}:")
        print(f"  all-races log loss: {s['log_loss']:.4f}  acc {s['accuracy']:.4f}  tau {s['kendall_tau']:.4f}")
        print(f"  market subset: market {mk.mean():.4f}  model {md.mean():.4f}  "
              f"diff {d.mean():+.4f} (se {d.std(ddof=1)/np.sqrt(len(d)):.4f})")
        # pool
        logm, logp, starts, winners = [], [], [0], []
        a_b = (1.0, 0.0); pool = []
        for i, (_, _, m_p, s_p, w) in enumerate(scored):
            if i >= 30 and i % 30 == 0:
                lm = np.concatenate(logm); lp = np.concatenate(logp)
                st = np.asarray(starts[:-1]); wn = np.asarray(winners)
                race_of = np.repeat(np.arange(len(st)), np.diff(starts))
                best = (np.inf, a_b)
                for a in np.linspace(0.6, 1.6, 11):
                    for b in np.linspace(0.0, 0.8, 9):
                        z = a * lm + b * lp
                        zmax = np.maximum.reduceat(z, st)
                        sums = np.add.reduceat(np.exp(z - zmax[race_of]), st)
                        loss = float(-np.mean(z[wn] - zmax - np.log(sums)))
                        if loss < best[0]:
                            best = (loss, (float(a), float(b)))
                a_b = best[1]
            a, b = a_b
            z = a * np.log(m_p) + b * np.log(s_p); z -= z.max()
            q = np.exp(z); q /= q.sum()
            pool.append(-math.log(q[w]))
            logm.append(np.log(m_p)); logp.append(np.log(s_p))
            winners.append(starts[-1] + w); starts.append(starts[-1] + len(m_p))
        print(f"  pool: {np.mean(pool):.4f}  final (a,b)=({a_b[0]:.2f},{a_b[1]:.2f})\n")


if __name__ == "__main__":
    main()
