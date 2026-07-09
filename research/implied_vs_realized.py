"""Implied vs realized volatility: does market-implied scale predict a
horse's FUTURE finish-time dispersion? (The options-market analogy.)

Career split per horse (6+ runs with both halves non-trivial): implied vol =
mean fitted scale over the FIRST half of its races; realized vol = sd of its
within-race-standardized finish-time residuals over the SECOND half (ability
removed by demeaning within the half). Cross-horse correlations:
    implied(1st)  -> realized(2nd)   the headline prediction test
    realized(1st) -> realized(2nd)   is realized vol a stable trait at all?
    implied(1st)  -> implied(2nd)    persistence (aggregate was 0.772)
Plus favorite/longshot split and a partial correlation controlling mean
market rank (the proxy-confound guard).

Run:  .venv/bin/python research/implied_vs_realized.py   (needs the npz)

Measured (July 2026, 2,979 horses with 8+ timed runs):
    implied(1st)->realized(2nd)   +0.068 sd / +0.097 MAD   (the headline)
    realized(1st)->realized(2nd)  +0.077 sd / +0.091 MAD   (the trait ceiling)
    implied(1st)->implied(2nd)    +0.464                    (persistence)
    implied->realized | mkt rank  +0.019                    (confound guard)
    front/back of market split:   +0.025 / +0.048
Verdicts: (1) per-horse time-volatility is genuinely almost not a stable
trait (ceiling ~0.08-0.09; MAD does not rescue it); (2) implied scale hits
that ceiling but adds NOTHING beyond market rank — the 0.46-0.77 persistence
is market-rank/class geometry, not horse temperament (the romantic reading
retracted); (3) reconciliation: loc+scale still improves place pricing at
every position — scale earns its keep as a per-race PRICING parameter
capturing rank-distribution shape, not as psychology; (4) the options
IV->RV analogy fails on the RV side: unlike equity vol, a handicapped
racehorse's residual variance is nearly serially independent.
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import zipfile
from collections import defaultdict

import numpy as np


def rebuild_times():
    """Replicate the loader's race ordering, attaching standardized times."""
    zip_path = os.path.expanduser("~/.cache/winning/hkracing.zip")
    races, runs = {}, defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                races[row["race_id"]] = row.get("date", "")
        with zf.open("runs.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                runs[row["race_id"]].append(row)

    dated = []
    for race_id, rows in runs.items():
        date = races.get(race_id)
        try:
            ordinal = datetime.date.fromisoformat(date).toordinal()
        except (TypeError, ValueError):
            continue
        entries = []
        ok_place = True
        for row in rows:
            try:
                pos = int(float(row["result"]))
                odds = float(row["win_odds"])
            except (KeyError, ValueError):
                continue
            try:
                po = float(row["place_odds"])
                if not po > 0:
                    ok_place = False
            except (KeyError, ValueError, TypeError):
                ok_place = False
            try:
                ft = float(row["finish_time"])
            except (KeyError, ValueError, TypeError):
                ft = None
            entries.append((f"horse_{row['horse_id']}", pos, ft))
        if len(entries) < 2 or min(e[1] for e in entries) != 1:
            continue
        if not (ok_place and len(entries) >= 4):
            continue  # mirrors the place_market filter
        dated.append((ordinal, entries))
    dated.sort(key=lambda t: t[0])
    return dated


def main():
    d = np.load(os.path.expanduser("~/.cache/winning/implied_params.npz"), allow_pickle=True)
    dated = rebuild_times()
    n_races = int(d["race"].max()) + 1
    assert n_races == len(dated), f"alignment: npz {n_races} vs rebuilt {len(dated)}"

    # standardized time residuals per race
    time_resid = {}  # (race_idx, horse) -> z
    for ridx, (_, entries) in enumerate(dated):
        ts = np.array([ft for _, _, ft in entries if ft is not None])
        if len(ts) < 4:
            continue
        med = np.median(ts)
        mad = np.median(np.abs(ts - med)) + 1e-9
        for nm, _, ft in entries:
            if ft is not None:
                time_resid[(ridx, nm)] = (ft - med) / mad

    # spot-check alignment: finish ranks must agree
    by_race_npz = defaultdict(dict)
    for r, h, fr in zip(d["race"], d["horse"], d["finish_rank"]):
        by_race_npz[int(r)][str(h)] = int(fr)
    mism = 0
    for ridx in list(by_race_npz)[:200]:
        for nm, pos, _ in dated[ridx][1]:
            if nm in by_race_npz[ridx] and by_race_npz[ridx][nm] != pos:
                mism += 1
    assert mism == 0, f"{mism} rank mismatches — ordering broke"

    # per-horse career arrays, chronological
    career = defaultdict(list)  # horse -> [(race_idx, scale, mrank, z_time)]
    for r, h, sc, mr in zip(d["race"], d["horse"], d["scale"], d["market_rank"]):
        z = time_resid.get((int(r), str(h)))
        career[str(h)].append((int(r), float(sc), int(mr), z))

    rows = []
    for h, lst in career.items():
        lst.sort()
        zs = [(sc, mr, z) for _, sc, mr, z in lst if z is not None]
        if len(zs) < 8:
            continue
        half = len(zs) // 2
        a, b = zs[:half], zs[half:]
        za = np.array([z for _, _, z in a]); zb = np.array([z for _, _, z in b])

        def robust(v):  # MAD-based: immune to one eased/catastrophic run
            return float(np.median(np.abs(v - np.median(v)))) if len(v) > 2 else None

        rows.append({
            "impl1": np.mean([s for s, _, _ in a]),
            "impl2": np.mean([s for s, _, _ in b]),
            "real1": za.std(ddof=1) if len(za) > 2 else None,
            "real2": zb.std(ddof=1) if len(zb) > 2 else None,
            "rob1": robust(za),
            "rob2": robust(zb),
            "mrank": np.mean([m for _, _, m, _ in lst]),
        })
    rows = [r for r in rows if r["real1"] is not None and r["real2"] is not None
            and r["rob1"] is not None and r["rob2"] is not None]
    print(f"horses with 8+ timed runs: {len(rows)}\n")

    def corr(x, y):
        return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])

    i1 = [r["impl1"] for r in rows]; i2 = [r["impl2"] for r in rows]
    r1 = [r["real1"] for r in rows]; r2 = [r["real2"] for r in rows]
    mk = [r["mrank"] for r in rows]
    b1 = [r["rob1"] for r in rows]; b2 = [r["rob2"] for r in rows]
    print(f"implied(1st) -> realized(2nd):  {corr(i1, r2):+.3f}   <- the headline (sd-based)")
    print(f"implied(1st) -> robust(2nd):    {corr(i1, b2):+.3f}   <- headline, MAD-based")
    print(f"realized(1st) -> realized(2nd): {corr(r1, r2):+.3f}   (trait ceiling, sd)")
    print(f"robust(1st) -> robust(2nd):     {corr(b1, b2):+.3f}   (trait ceiling, MAD)")
    print(f"implied(1st) -> implied(2nd):   {corr(i1, i2):+.3f}   (persistence)")
    print(f"realized(1st) -> implied(2nd):  {corr(r1, i2):+.3f}   (market learns from vol?)")

    # partial corr implied->realized controlling mean market rank
    def partial(x, y, z):
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return float(np.corrcoef(rx, ry)[0, 1])
    print(f"implied -> realized | mkt rank:  {partial(i1, r2, mk):+.3f}   (confound guard)")

    med = np.median(mk)
    fav = [i for i, r in enumerate(rows) if r["mrank"] <= med]
    lng = [i for i, r in enumerate(rows) if r["mrank"] > med]
    for label, idx in (("front-of-market horses", fav), ("back-of-market horses", lng)):
        print(f"  {label:24s} implied->realized: "
              f"{corr([i1[i] for i in idx], [r2[i] for i in idx]):+.3f}  (n={len(idx)})")


if __name__ == "__main__":
    main()
