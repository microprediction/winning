"""F1 with prices AND results: does the hybrid earn pool weight?

The pure results rater earned b = 0.00 against current F1 odds. This run
feeds the rater PAST races' market consensus as ability observations
(inverted through the ability transform, as in the HK market-hybrid) on top
of all results, then asks two questions on the 135 market-covered scored
races: (1) how much does odds-history close the fundamentals-to-market gap?
(2) does the hybrid earn nonzero weight in a prequential pool with the
current-race market?

Run:  .venv/bin/python research/f1_hybrid_market.py
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
from market_hybrid import MarketHybridThurstoneRating


class HybridEraSlab(MarketHybridThurstoneRating, EraSlabThurstoneRating):
    """Era-adaptive slab + past-odds observations (MRO does the work)."""


def main():
    market = defaultdict(dict)
    with open(os.path.expanduser("~/.cache/winning/f1_market.csv")) as f:
        for row in csv.DictReader(f):
            if row["race_key"] in EXCLUDE:
                continue
            year, slug = row["race_key"].split("_", 1)
            gp = GP_MAP.get(slug)
            if gp is None or "any-other" in norm_driver(row["driver"]):
                continue
            market[(int(year), gp)][norm_driver(row["driver"])] = float(row["consensus_prob"])

    with zipfile.ZipFile(os.path.expanduser("~/.cache/winning/f1db-csv.zip")) as zf:
        race_meta = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                race_meta[row["date"]] = (int(row["year"]), row["grandPrixId"])

    from f1_market_test import main as _unused  # noqa: F401  (for parity of env)
    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    # rebuild event dates exactly as in f1_market_test
    with zipfile.ZipFile(os.path.expanduser("~/.cache/winning/f1db-csv.zip")) as zf:
        rd = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rd[row["id"]] = row["date"]
        by_race = defaultdict(list)
        with zf.open("f1db-races-race-results.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                by_race[row["raceId"]].append(row)
    event_dates = []
    for race_id, rows in by_race.items():
        date = rd.get(race_id)
        if not date:
            continue
        seen = set()
        cnt = 0
        for row in sorted(rows, key=lambda r: int(r.get("positionDisplayOrder") or 0)):
            if row["driverId"] in seen:
                continue
            seen.add(row["driverId"])
            cnt += 1
        if cnt >= 2:
            event_dates.append(date)
    event_dates.sort()
    assert len(event_dates) == len(events)

    n_warm = int(len(events) * 0.2)
    system = HybridEraSlab(market_obs_sd=0.7)
    CLIP = 1e-9
    scored = []
    for idx, ev in enumerate(events):
        system.elapse(ev.dt)
        meta = race_meta.get(event_dates[idx])
        mk = market.get(meta) if meta else None
        if idx >= n_warm and mk:
            probs = np.asarray(system.win_probabilities(ev.names), dtype=float)
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
                    scored.append((-math.log(m_p[w]), -math.log(s_p[w]), m_p, s_p, w))
        system.observe(ev.names, ev.ranks, dt=0.0)
        # feed the odds of THIS race after observing it (past info for later races)
        if mk:
            names_in = [nm for nm in ev.names if nm in mk]
            if len(names_in) >= 8:
                probs_in = np.asarray([mk[nm] for nm in names_in], float)
                probs_in /= probs_in.sum()
                system.observe_market(names_in, list(probs_in))

    mk_ll = np.array([r[0] for r in scored]); hy_ll = np.array([r[1] for r in scored])
    d = hy_ll - mk_ll
    print(f"scored races: {len(scored)}")
    print(f"market:            {mk_ll.mean():.4f}")
    print(f"hybrid (odds+res): {hy_ll.mean():.4f}   [pure results was 2.1167]")
    print(f"hybrid - market:   {d.mean():+.4f} (se {d.std(ddof=1)/np.sqrt(len(d)):.4f})")

    # prequential pool
    logm, logp, starts, winners = [], [], [0], []
    a_b = (1.0, 0.0); pool_ll = []
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
        pool_ll.append(-math.log(q[w]))
        logm.append(np.log(m_p)); logp.append(np.log(s_p))
        winners.append(starts[-1] + w); starts.append(starts[-1] + len(m_p))
    print(f"pool:              {np.mean(pool_ll):.4f}   final (a,b)=({a_b[0]:.2f},{a_b[1]:.2f})")


if __name__ == "__main__":
    main()
