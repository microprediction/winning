"""The Halo 2 beta matchmaking data from the original TrueSkill paper
(Herbrich, Minka & Graepel, NIPS 2006), as distributed with the Model-Based
Machine Learning book (dotnet/mbmlbook, MIT).

Two variants:
- head-to-head: 6,028 two-player Slayer games with scores (draws possible)
- free-for-all: multiplayer games ranked by score — the multi-entrant setting
  the lattice rater targets, on TrueSkill's own published data

Score attributes are omitted from the XML when zero.
"""

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET
from typing import List

from .events import Event
from .fetching import fetch

_BASE = (
    "https://raw.githubusercontent.com/dotnet/mbmlbook/main/"
    "src/3.%20Meeting%20Your%20Match/Data/"
)
_FILES = {
    "head-to-head": "Halo2-HeadToHead.objml",
    "free-for-all": "Halo2-FreeForAll.objml",
}


def _stamp(s: str) -> float:
    return datetime.datetime.strptime(s, "%m/%d/%Y %H:%M:%S").timestamp()


def halo2_events(mode: str = "head-to-head") -> List[Event]:
    fname = _FILES[mode]
    path = fetch(_BASE + fname.replace(" ", "%20"), fname, timeout=600)
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for game in root.iter():
        tag = game.tag.rsplit("}", 1)[-1]
        if tag == "TwoPlayerGame":
            p1, p2 = game.get("Player1"), game.get("Player2")
            s1 = float(game.get("Player1Score", 0.0))
            s2 = float(game.get("Player2Score", 0.0))
            end = game.get("EndTime")
            if not (p1 and p2 and end and p1 != p2):
                continue
            if s1 > s2:
                ranks = [1, 2]
            elif s2 > s1:
                ranks = [2, 1]
            else:
                ranks = [1, 1]
            rows.append((_stamp(end), [p1, p2], ranks))
        elif tag == "MultiPlayerGame":
            end = game.get("EndTime")
            scores = _player_scores(game)
            if not end or len(scores) < 2:
                continue
            names = list(scores)
            order = sorted(names, key=lambda nm: -scores[nm])
            ranks_by_name = {}
            rank = 0
            prev_score = None
            for pos, nm in enumerate(order, start=1):
                if scores[nm] != prev_score:
                    rank = pos
                    prev_score = scores[nm]
                ranks_by_name[nm] = rank
            rows.append((_stamp(end), names, [ranks_by_name[nm] for nm in names]))

    rows.sort(key=lambda r: r[0])
    events: List[Event] = []
    prev = None
    for stamp, names, ranks in rows:
        dt = 0.0 if prev is None else max(0.0, (stamp - prev) / 86400.0)  # days
        prev = stamp
        events.append(Event(names=names, ranks=ranks, dt=dt))
    return events


def _player_scores(game) -> dict:
    """PlayerScores is a key/value dictionary: <x:key>Gamer..</x:key><Int32>..</Int32>."""
    scores = {}
    key = None
    for node in game.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "key" or tag.endswith("Key"):
            key = (node.text or "").strip()
        elif tag == "Int32" and key:
            try:
                scores[key] = float((node.text or "0").strip())
            except ValueError:
                scores[key] = 0.0
            key = None
    return scores


def dataset(args) -> tuple:
    events = halo2_events("head-to-head")
    n_draws = sum(1 for ev in events if ev.ranks[0] == ev.ranks[1])
    return events, (
        f"Halo 2 beta head-to-head ({len(events)} games, {n_draws} draws; "
        "the original TrueSkill paper's data, mbmlbook, MIT)"
    )


def dataset_ffa(args) -> tuple:
    events = halo2_events("free-for-all")
    sizes = [len(ev.names) for ev in events]
    return events, (
        f"Halo 2 beta free-for-all ({len(events)} games, field sizes "
        f"{min(sizes)}-{max(sizes)}; the original TrueSkill paper's data, mbmlbook, MIT)"
    )
