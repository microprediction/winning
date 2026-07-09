"""Do ANY rating systems add power to the pre-qualifying market? (88 races)

For each system: prequential pool q ∝ market^a · model^b on the PRE stratum
(snapshots two or more days before the race), exponents refit every 20
scored races on past races only. Systems include the grid-aware rater,
which holds information the pre-qualifying market cannot have (qualifying
had not happened at snapshot time) — any weight it earns is timing
information, the way sharp bettors exploit early prices.

Run:  .venv/bin/python research/f1_pre_quali_pool.py

Measured (July 2026, 88 PRE races; market alone 1.5810):
    every system fits pool weight b = 0.00 (OpenSkill 0.10, noise), and the
    fitted temperature a = 1.20 overfits at this sample size, leaving every
    pool at 1.5993, worse than the raw market. The grid-aware rater, holding
    the realized qualifying result that the pre-qualifying snapshots
    provably lack, also earns zero weight: the market's car-form knowledge
    subsumes even the future grid. Companion stratification
    (f1_market_delineated.py): PRE n=88 market 1.581 vs results-only 2.177
    (+0.596, se 0.081) and vs grid-aware 1.926; SAT n=26 +0.199 (se 0.113)
    and POST n=21 +0.315 (se 0.141) tier-matched. The market's moat is
    mostly current car form, not qualifying contamination.
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
from winning import EloRating, Glicko2Rating
from winning.shims import OpenSkillRating, TrueSkillRating

CLIP = 1e-9


def main():
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

    def is_pre(key):
        ts = datetime.datetime.fromisoformat(snap_ts[key]).date()
        rd = datetime.date.fromisoformat(race_date[key])
        return (rd - ts).days >= 2

    dates = rebuild_dates()
    grid = load_grid()

    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    n_warm = int(len(events) * 0.2)
    systems = {
        "Thurstone (results)": EraSlabThurstoneRating(),
        "Thurstone (grid-aware)": EraSlabThurstoneRating(),
        "TrueSkill": TrueSkillRating(),
        "Elo": EloRating(),
        "Glicko-2": Glicko2Rating(),
        "OpenSkill PL": OpenSkillRating("PlackettLuce"),
    }
    collected = {nm: [] for nm in systems}  # (m_p, s_p, w) per PRE race
    for idx, ev in enumerate(events):
        for sy in systems.values():
            sy.elapse(ev.dt)
        date = dates[idx]
        gmap = grid.get(date, {})
        if len([1 for nm in ev.names if nm in gmap]) >= max(2, len(ev.names) // 2):
            systems["Thurstone (grid-aware)"].observe(
                ev.names, grid_ranks_for(ev, gmap), dt=0.0)
        meta = race_meta.get(date)
        mk = market.get(meta) if meta else None
        if idx >= n_warm and mk and is_pre(meta):
            matched = [i for i, nm in enumerate(ev.names) if nm in mk]
            if len(matched) >= 8:
                winner = min(range(len(ev.ranks)), key=lambda i: ev.ranks[i])
                if winner in matched:
                    w = matched.index(winner)
                    m_p = np.maximum(np.asarray([mk[ev.names[i]] for i in matched]), CLIP)
                    m_p /= m_p.sum()
                    for nm, sy in systems.items():
                        probs = np.asarray(sy.win_probabilities(ev.names), dtype=float)
                        s_p = np.maximum(np.asarray([probs[i] for i in matched]), CLIP)
                        s_p /= s_p.sum()
                        collected[nm].append((m_p, s_p, w))
        for sy in systems.values():
            sy.observe(ev.names, ev.ranks, dt=0.0)

    n = len(next(iter(collected.values())))
    print(f"PRE races scored: {n}\n")
    print(f"{'system':24s} {'model ll':>9s} {'pool ll':>9s} {'vs mkt (se)':>14s} {'final a,b':>11s}")
    mk_ll = None
    for nm, rows in collected.items():
        logm, logp, starts, winners = [], [], [0], []
        a_b = (1.0, 0.0)
        pool_ll, mkt_ll, mdl_ll = [], [], []
        for i, (m_p, s_p, w) in enumerate(rows):
            if i >= 20 and i % 20 == 0:
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
            pool_ll.append(-math.log(q[w]))
            mkt_ll.append(-math.log(m_p[w]))
            mdl_ll.append(-math.log(s_p[w]))
            logm.append(np.log(m_p)); logp.append(np.log(s_p))
            winners.append(starts[-1] + w); starts.append(starts[-1] + len(m_p))
        d = np.asarray(pool_ll) - np.asarray(mkt_ll)
        mk_ll = np.mean(mkt_ll)
        print(f"{nm:24s} {np.mean(mdl_ll):9.4f} {np.mean(pool_ll):9.4f} "
              f"{d.mean():+9.4f} ({d.std(ddof=1)/np.sqrt(len(d)):.4f}) "
              f"({a_b[0]:.2f},{a_b[1]:.2f})")
    print(f"\nmarket alone: {mk_ll:.4f}   (negative diff = pool improves on market)")


if __name__ == "__main__":
    main()
