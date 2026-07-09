"""Lichess open database, January 2013 (CC0): ~121k rated standard games.

Draws are real ties. Each game carries the players' Lichess ratings at game
time — the site's own production Glicko-2, trained on each player's entire
history — which we surface through the Market row (Elo-curve win probability
conditional on a decisive result). That baseline sees far more history than
the systems under test, so treat it as a ceiling like bookmaker odds.

Decompression uses the stdlib compression.zstd (Python 3.14+) or the
`zstandard` package if installed.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from .events import Event
from .fetching import fetch

_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def _open_zst(path: str):
    try:
        from compression.zstd import ZstdFile  # Python 3.14+

        return ZstdFile(path)
    except ImportError:
        try:
            import zstandard
        except ImportError as e:
            raise ImportError(
                "Reading Lichess data needs Python 3.14+ (stdlib compression.zstd) "
                "or `pip install zstandard`."
            ) from e
        return zstandard.open(path)


def chess_events(month: str = "2013-01", months=None) -> List[Event]:
    """One month by default; pass months=("2013-01", "2013-02", ...) to
    concatenate several (rows are re-sorted globally by timestamp)."""
    events: List[Event] = []
    rows = []
    for m in months if months is not None else (month,):
        path = fetch(_URL.format(month=m), f"lichess_{m}.pgn.zst", timeout=600)
        game = {}
        with _open_zst(path) as f:
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("["):
                    key, _, rest = line[1:-1].partition(" ")
                    game[key] = rest.strip('"')
                elif not line and game.get("Result") in ("1-0", "0-1", "1/2-1/2"):
                    row = _to_row(game)
                    if row:
                        rows.append(row)
                    game = {}

    rows.sort(key=lambda r: r[0])
    prev = None
    for stamp, white, black, ranks, market in rows:
        dt = 0.0 if prev is None else max(0.0, (stamp - prev) / 86400.0)  # days
        prev = stamp
        events.append(Event(names=[white, black], ranks=ranks, dt=dt, market=market))
    return events


def _to_row(game: dict) -> Optional[tuple]:
    white, black = game.get("White"), game.get("Black")
    if not (white and black and white != black):
        return None
    try:
        stamp = (
            datetime.datetime.strptime(
                game.get("UTCDate", "") + " " + game.get("UTCTime", "00:00:00"),
                "%Y.%m.%d %H:%M:%S",
            )
            .replace(tzinfo=datetime.timezone.utc)  # naive .timestamp() would
            .timestamp()  # be non-monotone across local DST transitions
        )
    except ValueError:
        return None
    ranks = {"1-0": [1, 2], "0-1": [2, 1], "1/2-1/2": [1, 1]}[game["Result"]]
    market = None
    try:
        we, be = float(game["WhiteElo"]), float(game["BlackElo"])
        p = 1.0 / (1.0 + 10.0 ** ((be - we) / 400.0))
        market = [p, 1.0 - p]
    except (KeyError, ValueError):
        pass
    return (stamp, white, black, ranks, market)


def dataset(args) -> tuple:
    events = chess_events()
    n_draws = sum(1 for ev in events if ev.ranks[0] == ev.ranks[1])
    return events, (
        f"Lichess rated standard chess, 2013-01 ({len(events)} games, {n_draws} draws; "
        "CC0; Market row = the site's own ratings at game time)"
    )
