"""English Premier League matches from football-data.co.uk (free CSVs).

Clubs are the contestants (a 1-D club rating; no lineup data). Draws are real
here: they enter as tied ranks — every system updates on them (Elo/Glicko score
0.5, the lattice treats them as dead heats) — but drawn matches are not scored
by the winner metrics (see forward_chain), which is stated in the README.

Bookmaker odds (Bet365 home/draw/away) provide a Market ceiling row: the
market's win probabilities conditional on a decisive result, comparable to the
rating systems' two-contestant win probabilities.
"""

from __future__ import annotations

import csv
import datetime
import io
import os
from typing import List

from .events import Event
from .fetching import fetch

_URL = os.environ.get(
    "WINNING_FOOTBALL_URL", "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
)


def _fetch_season(season: str) -> str:
    path = fetch(_URL.format(season=season), f"E0_{season}.csv")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def epl_events(first_season: int = 2005, last_season: int = 2024) -> List[Event]:
    """Seasons are named by their starting year (2005 -> '0506')."""
    rows = []
    for year in range(first_season, last_season + 1):
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        text = _fetch_season(season)
        for row in csv.DictReader(io.StringIO(text)):
            home, away, result = row.get("HomeTeam"), row.get("AwayTeam"), row.get("FTR")
            date = _parse_date(row.get("Date") or "")
            if not (home and away and result in ("H", "D", "A") and date):
                continue
            market = None
            try:
                ph, pd_, pa = (
                    1.0 / float(row["B365H"]),
                    1.0 / float(row["B365D"]),
                    1.0 / float(row["B365A"]),
                )
                # win probabilities conditional on a decisive result, overround removed
                market = [ph / (ph + pa), pa / (ph + pa)]
                del pd_
            except (KeyError, ValueError, ZeroDivisionError):
                pass
            ranks = {"H": [1, 2], "A": [2, 1], "D": [1, 1]}[result]
            rows.append((date.toordinal(), home, away, ranks, market))

    rows.sort(key=lambda r: r[0])
    events: List[Event] = []
    prev = None
    for ordinal, home, away, ranks, market in rows:
        dt = 0.0 if prev is None else float(ordinal - prev)
        prev = ordinal
        events.append(Event(names=[home, away], ranks=ranks, dt=dt, market=market))
    return events


def dataset(args) -> tuple:
    events = epl_events()
    n_draws = sum(1 for ev in events if ev.ranks[0] == ev.ranks[1])
    title = (
        f"English Premier League 2005-2025 ({len(events)} matches, {n_draws} draws; "
        "football-data.co.uk, Bet365 odds as market ceiling)"
    )
    return events, title
