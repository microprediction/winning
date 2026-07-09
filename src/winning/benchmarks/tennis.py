"""ATP and WTA tennis events from Jeff Sackmann's match archives (CC BY-NC-SA 4.0).

Fetched at run time and cached locally; not vendored into the repo. Tennis is
the N=2 longitudinal case: real players, real dates, ~3k matches per tour-year.

The original JeffSackmann repositories are no longer available on GitHub
(checked July 2026), so the defaults point at fork mirrors carrying data
through 2024; override with the WINNING_ATP_URL / WINNING_WTA_URL environment
variables (format strings with a {year} placeholder).
"""

from __future__ import annotations

import csv
import io
import os
from typing import List

from .events import Event
from .fetching import cache_dir as _cache_dir_impl
from .fetching import fetch

_URLS = {
    "atp": os.environ.get(
        "WINNING_ATP_URL",
        "https://raw.githubusercontent.com/VictorSquidWei/tennis_atp/master/atp_matches_{year}.csv",
    ),
    "wta": os.environ.get(
        "WINNING_WTA_URL",
        "https://raw.githubusercontent.com/VictorSquidWei/tennis_wta/master/wta_matches_{year}.csv",
    ),
}


def _cache_dir() -> str:
    return _cache_dir_impl()


def _fetch_year(year: int, tour: str = "atp") -> str:
    path = fetch(_URLS[tour].format(year=year), f"{tour}_matches_{year}.csv")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tennis_events(tour: str = "atp", start_year: int = 2013, end_year: int = 2024) -> List[Event]:
    rows = []
    for year in range(start_year, end_year + 1):
        text = _fetch_year(year, tour=tour)
        for row in csv.DictReader(io.StringIO(text)):
            date = row.get("tourney_date") or ""
            w, l = row.get("winner_name"), row.get("loser_name")
            if not (date.isdigit() and w and l and w != l):
                continue
            rows.append((int(date), w, l))
    rows.sort(key=lambda r: r[0])

    events: List[Event] = []
    prev_ordinal = None
    for date, w, l in rows:
        ordinal = _date_ordinal(date)
        dt = 0.0 if prev_ordinal is None else max(0.0, float(ordinal - prev_ordinal))
        prev_ordinal = ordinal
        events.append(Event(names=[w, l], ranks=[1, 2], dt=dt))
    return events


def _date_ordinal(yyyymmdd: int) -> int:
    import datetime

    y, m, d = yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100
    try:
        return datetime.date(y, m, d).toordinal()
    except ValueError:
        return datetime.date(y, 1, 1).toordinal()
