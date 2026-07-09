"""F1 vs the betting market: the verdict on 144 archived races.

Joins the Oddschecker consensus (research/f1_odds_parse.py) to f1db races,
runs the era-adaptive slab rater prequentially over ALL grands prix, and on
market-covered races past warmup scores market vs model vs uniform on the
matched-driver subset (both renormalized over the same drivers; races where
the actual winner is unmatched are skipped). Includes a tempered pool
(market^a * model^b, prequential refit) with the honest caveat that ~130
scored races give wide error bars on b.

Known data caveats (see f1_odds_parse.py report): some snapshots are
post-qualifying (market embeds grid info); 2025_australian is a pre-season
antepost market and is excluded.

Run:  .venv/bin/python research/f1_market_test.py

Measured (July 2026, 135 scored races 2008-2025):
    market 1.5466 | pool 1.5382 (a=1.10, b=0.00) | model 2.1167 | uniform 3.0182
    model - market = +0.5701 (se 0.065)
Verdict: the F1 win market is ~0.57 log loss ahead of the best fundamental
rater (a larger moat than HK's 0.35) and the fitted pool gives ratings zero
weight — replicated market efficiency on a second sport. Post-qualifying
snapshots embed grid position: the market prices the present. The only
wrinkle is a=1.10 mild favorite underconfidence worth 0.008 — inside any
margin. The formula-edge route (derivative markets priced off win odds via
lossy Harville transforms) remains untested for want of derivative odds.
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

GP_MAP = {
    "abu-dhabi-grand-prix": "abu-dhabi", "australian-grand-prix": "australia",
    "austrian-grand-prix": "austria", "bahrain-grand-prix": "bahrain",
    "belgian-grand-prix": "belgium", "brazilian-grand-prix": "brazil",
    "british-grand-prix": "great-britain", "canadian-grand-prix": "canada",
    "chinese-grand-prix": "china", "dutch-grand-prix": "netherlands",
    "emilia-romagna-grand-prix": "emilia-romagna", "european-grand-prix": "europe",
    "french-grand-prix": "france", "german-grand-prix": "germany",
    "hungarian-grand-prix": "hungary", "indian-grand-prix": "india",
    "italian-grand-prix": "italy", "japanese-grand-prix": "japan",
    "korean-grand-prix": "south-korea", "malaysian-grand-prix": "malaysia",
    "mexican-grand-prix": "mexico", "monaco-grand-prix": "monaco",
    "russian-grand-prix": "russia", "sakhir-grand-prix": "sakhir",
    "saudi-arabia-grand-prix": "saudi-arabia", "singapore-grand-prix": "singapore",
    "spanish-grand-prix": "spain", "styrian-grand-prix": "styria",
    "turkish-grand-prix": "turkey", "united-states-grand-prix": "united-states",
    "us-grand-prix": "united-states",
}
EXCLUDE = {"2025_australian-grand-prix"}  # pre-season antepost


def norm_driver(s: str) -> str:
    s = s.split("(")[0].strip()
    if "," in s:
        last, first = [t.strip() for t in s.split(",", 1)]
        s = f"{first} {last}"
    s = s.lower().replace(".", "").replace("'", "")
    fixes = {"felippe massa": "felipe massa", "kimi raikkonen": "kimi raikkonen"}
    s = fixes.get(s, s)
    return "-".join(s.split())


def main():
    # market data
    market = defaultdict(dict)  # (year, gpId) -> {driver_slug: prob}
    with open(os.path.expanduser("~/.cache/winning/f1_market.csv")) as f:
        for row in csv.DictReader(f):
            if row["race_key"] in EXCLUDE:
                continue
            year, slug = row["race_key"].split("_", 1)
            gp = GP_MAP.get(slug)
            if gp is None or "any-other" in norm_driver(row["driver"]):
                continue
            market[(int(year), gp)][norm_driver(row["driver"])] = float(row["consensus_prob"])

    # f1db: raceId -> (year, gpId); driverId set for name matching
    with zipfile.ZipFile(os.path.expanduser("~/.cache/winning/f1db-csv.zip")) as zf:
        race_meta = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                race_meta[row["date"]] = (int(row["year"]), row["grandPrixId"])

    # events in loader order carry no ids; rebuild the same date ordering
    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    # loader sorted by date and dates are unique; recover each event's date order
    dates = sorted(race_meta)
    # f1_events dropped races missing dates/entrants; align by sequential date match:
    # walk events and dates together using field sizes as a checksum-free join
    # (dates list includes ALL races; events subset). We instead rebuild dates
    # from the results file the same way the loader did:
    import winning.benchmarks.f1 as f1mod
    import datetime as _dt

    with zipfile.ZipFile(os.path.expanduser("~/.cache/winning/f1db-csv.zip")) as zf:
        rd = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rd[row["id"]] = row["date"]
        from collections import defaultdict as dd
        by_race = dd(list)
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
            drv = row["driverId"]
            if drv in seen:
                continue
            seen.add(drv)
            cnt += 1
        if cnt >= 2:
            event_dates.append(date)
    event_dates.sort()
    assert len(event_dates) == len(events), (len(event_dates), len(events))

    n_warm = int(len(events) * 0.2)
    system = EraSlabThurstoneRating()
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
                    u = np.full(len(matched), 1.0 / len(matched))
                    scored.append((meta, len(matched),
                                   -math.log(m_p[w]), -math.log(s_p[w]), -math.log(u[w]),
                                   m_p, s_p, w))
        system.observe(ev.names, ev.ranks, dt=0.0)

    print(f"scored races with market coverage: {len(scored)}")
    mk_ll = np.array([r[2] for r in scored]); md_ll = np.array([r[3] for r in scored])
    un_ll = np.array([r[4] for r in scored])
    d = md_ll - mk_ll
    print(f"market log loss:  {mk_ll.mean():.4f}")
    print(f"model  log loss:  {md_ll.mean():.4f}   (era-adaptive slab)")
    print(f"uniform:          {un_ll.mean():.4f}")
    print(f"model - market:   {d.mean():+.4f}  (se {d.std(ddof=1)/np.sqrt(len(d)):.4f})")

    # tempered pool, prequential, refit every 30 scored races
    logm_all, logp_all, starts, winners = [], [], [0], []
    a_b = (1.0, 0.0)
    pool_ll = []
    for i, (_, n, _, _, _, m_p, s_p, w) in enumerate(scored):
        if i >= 30 and i % 30 == 0:
            lm = np.concatenate(logm_all); lp = np.concatenate(logp_all)
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
        z = a * np.log(m_p) + b * np.log(s_p)
        z -= z.max()
        q = np.exp(z); q /= q.sum()
        pool_ll.append(-math.log(q[w]))
        logm_all.append(np.log(m_p)); logp_all.append(np.log(s_p))
        winners.append(starts[-1] + w); starts.append(starts[-1] + len(m_p))
    print(f"tempered pool:    {np.mean(pool_ll):.4f}   final (a,b)=({a_b[0]:.2f},{a_b[1]:.2f})"
          f"   [~{len(scored)} races: wide error bars]")


if __name__ == "__main__":
    main()
