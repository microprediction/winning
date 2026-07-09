"""Formula 1 race results from the f1db project (CC-BY-4.0).

One bulk CSV zip, self-updating via the GitHub latest-release redirect. Every
World Championship race since 1950: ~20-driver grids with full finish orders
and persistent driver identities — the flagship real multi-entrant dataset.

Retirements (positionText DNF etc.) are ranked as tied last: the finish order
records that they lost to every classified finisher, and nothing more. That
charges drivers for mechanical failures; a status-aware treatment is future
work.
"""

from __future__ import annotations

import csv
import datetime
import io
import zipfile
from collections import defaultdict
from typing import List

from .events import Event
from .fetching import fetch

_ZIP_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"


def _read_csv(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))


def f1_events(first_year: int = 1950, last_year: int = 2100) -> List[Event]:
    path = fetch(_ZIP_URL, "f1db-csv.zip", timeout=300)
    with zipfile.ZipFile(path) as zf:
        race_date = {}
        for row in _read_csv(zf, "f1db-races.csv"):
            race_date[row["id"]] = row["date"]

        by_race = defaultdict(list)
        for row in _read_csv(zf, "f1db-races-race-results.csv"):
            if not (first_year <= int(row["year"]) <= last_year):
                continue
            by_race[row["raceId"]].append(row)

    events: List[Event] = []
    dated = []
    for race_id, rows in by_race.items():
        date = race_date.get(race_id)
        if not date:
            continue
        ordinal = datetime.date.fromisoformat(date).toordinal()
        names, ranks = [], []
        worst = 0
        finishers, dnfs = [], []
        seen = set()  # 1950s shared drives: one driver, several cars, one race
        for row in sorted(rows, key=lambda r: int(r.get("positionDisplayOrder") or 0)):
            driver = row["driverId"]
            if driver in seen:
                continue
            seen.add(driver)
            pos = row.get("positionNumber")
            if pos:
                finishers.append((int(pos), driver))
                worst = max(worst, int(pos))
            else:
                dnfs.append(driver)
        if len(finishers) + len(dnfs) < 2:
            continue
        for pos, driver in sorted(finishers):
            names.append(driver)
            ranks.append(pos)
        for driver in dnfs:
            names.append(driver)
            ranks.append(worst + 1)  # tied last
        dated.append((ordinal, Event(names=names, ranks=ranks)))

    dated.sort(key=lambda t: t[0])
    prev = None
    for ordinal, ev in dated:
        ev.dt = 0.0 if prev is None else float(ordinal - prev)
        prev = ordinal
        events.append(ev)
    return events


def dataset(args) -> tuple:
    events = f1_events()
    return events, (
        f"Formula 1 1950-present ({len(events)} grands prix; f1db, CC-BY-4.0; "
        "DNFs tied last)"
    )
