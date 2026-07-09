"""Kaggle-hosted datasets: Hong Kong horse racing and PUBG finish placements.

Both need Kaggle API credentials in ~/.kaggle/kaggle.json (and the PUBG
competition additionally requires accepting its rules on the Kaggle site).

- gdaley/hkracing: ~6.3k Sha Tin/Happy Valley races 1997-2005 with NAMED
  horses AND win odds — rating systems vs the pari-mutuel market on the
  package's namesake sport (races.csv + runs.csv).
- pubg-finish-placement-prediction: ~90-group free-for-all matches, the
  standard large-field FFA benchmark (train_V2.csv, winPlacePerc).
"""

from __future__ import annotations

import csv
import datetime
import os
import zipfile
from collections import defaultdict
from typing import List

from .events import Event
from .fetching import cache_dir


def _require_kaggle():
    has_json = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_json or has_env):
        raise RuntimeError(
            "Kaggle credentials not found. Save an API token to "
            "~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_USERNAME and "
            "KAGGLE_KEY. The PUBG competition also requires accepting its "
            "rules on the Kaggle website."
        )


def _kaggle_download(kind: str, ref: str, filename: str) -> str:
    """kind is 'datasets' or 'competitions'; returns the local zip/csv path."""
    out = os.path.join(cache_dir(), filename)
    if not os.path.exists(out):
        _require_kaggle()
        import subprocess

        flag = "-d" if kind == "datasets" else "-c"
        subprocess.run(
            ["kaggle", kind, "download", flag, ref, "-p", cache_dir()], check=True
        )
        if not os.path.exists(out):
            raise RuntimeError(
                f"kaggle download finished but {out} is missing; check the CLI "
                "output above (a partial file at that path must be deleted)"
            )
    return out


def hkracing_events(oracle_temperature: float = None) -> List[Event]:
    """oracle_temperature: if set, attach truth ∝ market^a to every event —
    the recalibrated pari-mutuel as a real-data oracle (the 'rating lab' of
    planning/rating_lab.md; a=1.05 was fit leakage-free by
    research/beat_the_market.py). Systems are then scored by tv_vs_oracle,
    comparing probability vectors instead of single-draw outcomes."""
    zip_path = _kaggle_download("datasets", "gdaley/hkracing", "hkracing.zip")
    races, runs = {}, defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        import io

        with zf.open("races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                races[row["race_id"]] = (
                    row.get("date", ""),
                    {
                        "distance": row.get("distance"),
                        "venue": row.get("venue"),
                        "surface": row.get("surface"),
                        "going": row.get("going"),
                        "sec_time1": row.get("sec_time1"),
                    },
                )
        with zf.open("runs.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                runs[row["race_id"]].append(row)

    dated = []
    for race_id, rows in runs.items():
        date, context = races.get(race_id, (None, None))
        try:
            ordinal = datetime.date.fromisoformat(date).toordinal()
        except (TypeError, ValueError):
            continue
        names, ranks, inv_odds, inv_place = [], [], [], []
        for row in rows:
            try:
                pos = int(float(row["result"]))
                odds = float(row["win_odds"])
            except (KeyError, ValueError):
                continue
            names.append(f"horse_{row['horse_id']}")
            ranks.append(pos)
            inv_odds.append(1.0 / odds if odds > 0 else 0.0)
            try:
                po = float(row["place_odds"])
                inv_place.append(1.0 / po if po > 0 else None)
            except (KeyError, ValueError, TypeError):
                inv_place.append(None)
        if len(names) < 2 or min(ranks) != 1:
            continue
        total = sum(inv_odds)
        market = [q / total for q in inv_odds] if total > 0 else None
        place_market = None
        if all(q is not None for q in inv_place) and len(inv_place) >= 4:
            n_places = 3 if len(names) >= 7 else 2
            tot = sum(inv_place)
            if tot > 0:
                place_market = {
                    "probs": [q / tot * n_places for q in inv_place],
                    "n_places": n_places,
                }
        truth = None
        if oracle_temperature is not None and market is not None:
            powered = [q**oracle_temperature for q in market]
            z = sum(powered)
            truth = [q / z for q in powered]
        ctx = dict(context or {})
        if place_market is not None:
            ctx["place_market"] = place_market
        dated.append(
            (ordinal, Event(names=names, ranks=ranks, market=market, truth=truth, context=ctx))
        )

    dated.sort(key=lambda t: t[0])
    events: List[Event] = []
    prev = None
    for ordinal, ev in dated:
        ev.dt = 0.0 if prev is None else float(ordinal - prev)
        prev = ordinal
        events.append(ev)
    return events


def hkracing(args) -> tuple:
    events = hkracing_events()
    return events, (
        f"Hong Kong horse racing 1997-2005 ({len(events)} races, named horses; "
        "Kaggle gdaley/hkracing; Market row = pari-mutuel win odds)"
    )


def hkracing_lab(args) -> tuple:
    events = hkracing_events(oracle_temperature=1.05)
    return events, (
        f"HK racing LAB ({len(events)} races): oracle = market^1.05 (fitted "
        "recalibration); fundamental systems judged by TV to market-implied truth"
    )


def pubg_events(max_matches: int = 20000) -> List[Event]:
    zip_path = _kaggle_download(
        "competitions", "pubg-finish-placement-prediction", "pubg-finish-placement-prediction.zip"
    )
    matches = defaultdict(dict)  # matchId -> groupId -> winPlacePerc
    with zipfile.ZipFile(zip_path) as zf:
        import io

        with zf.open("train_V2.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                if row.get("matchType") != "solo":
                    continue
                try:
                    matches[row["matchId"]][row["Id"]] = float(row["winPlacePerc"])
                except (KeyError, ValueError):
                    continue

    events: List[Event] = []
    for match_id, placements in matches.items():
        if len(events) >= max_matches:
            break
        if len(placements) < 2:
            continue
        names = list(placements)
        order = sorted(names, key=lambda nm: -placements[nm])
        ranks_by = {}
        rank = 0
        prev = None
        for pos, nm in enumerate(order, start=1):
            if placements[nm] != prev:
                rank = pos
                prev = placements[nm]
            ranks_by[nm] = rank  # equal winPlacePerc is a tie, not dict order
        events.append(Event(names=names, ranks=[ranks_by[nm] for nm in names]))
    return events


def pubg(args) -> tuple:
    events = pubg_events()
    sizes = [len(ev.names) for ev in events]
    return events, (
        f"PUBG solo matches ({len(events)} matches, field sizes {min(sizes)}-{max(sizes)}; "
        "Kaggle competition data)"
    )
