"""Professional sumo bouts 1983-2021 (~110k after dedup), Makuuchi + Juryo.

Source: a GitHub mirror of the SumoDB-derived data.world dump (research use;
no formal license). Each bout appears twice in the raw file (once per rikishi
perspective); we keep one copy per unordered pair. Dense schedules — 15 bouts
per rikishi per basho, six basho per year — with real entry/exit and aging
dynamics, which exercises the time-diffusion machinery hard.
"""

from __future__ import annotations

import csv
import datetime
from typing import List

from .events import Event
from .fetching import fetch

_URL = (
    "https://raw.githubusercontent.com/ElVejigante/Sumo-Tournament-Predictor/"
    "master/dev%20files/data/source%20files/results.csv"
)


def sumo_events() -> List[Event]:
    path = fetch(_URL, "sumo_results.csv", timeout=300)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                year, month = (int(x) for x in row["basho"].split("."))
                day = int(row["day"])
                r1, r2 = row["rikishi1_id"], row["rikishi2_id"]
                win1 = int(row["rikishi1_win"])
            except (KeyError, ValueError):
                continue
            # every bout appears twice, once per rikishi perspective; keep the
            # canonical perspective (r1 < r2) so playoff re-matches on the same
            # day survive (a seen-set would silently drop them)
            if r1 == r2 or not int(r1) < int(r2):
                continue
            ordinal = datetime.date(year, month, 1).toordinal() + day - 1
            # rikishi_id is the stable identity; shikona (ring names) change
            name1, name2 = f"rikishi_{r1}", f"rikishi_{r2}"
            ranks = [1, 2] if win1 == 1 else [2, 1]
            rows.append((ordinal, name1, name2, ranks))

    rows.sort(key=lambda r: r[0])
    events: List[Event] = []
    prev = None
    for ordinal, n1, n2, ranks in rows:
        dt = 0.0 if prev is None else float(ordinal - prev)
        prev = ordinal
        events.append(Event(names=[n1, n2], ranks=ranks, dt=dt))
    return events


def dataset(args) -> tuple:
    events = sumo_events()
    return events, (
        f"Sumo bouts 1983-2021 ({len(events)} bouts, Makuuchi+Juryo; "
        "SumoDB via data.world mirror, research use)"
    )
