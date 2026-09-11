"""Build the cached Lichess header table the chess experiments read.

One month of the Lichess open database (database.lichess.org, CC0),
streamed and reduced to headers only: ~380MB of PGN becomes a 1MB
parquet. Idempotent; skips the download if the cache exists.

Run:  python research/chess/loader.py [YYYY-MM]
"""
from __future__ import annotations
import collections, io, os, re, sys, urllib.request
import pandas as pd, zstandard

MONTH = sys.argv[1] if len(sys.argv) > 1 else "2013-01"
CACHE = os.path.expanduser(
    f"~/.cache/winning/lichess_{MONTH.replace('-', '_')}_headers.parquet")
URL = (f"https://database.lichess.org/standard/"
       f"lichess_db_standard_rated_{MONTH}.pgn.zst")

def main():
    if os.path.exists(CACHE):
        print(f"cached already: {CACHE}")
        return
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"fetching {URL}")
    raw = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "winning-research"}),
        timeout=900).read()
    print(f"  {len(raw)/1e6:.1f} MB compressed; streaming headers")
    reader = io.TextIOWrapper(
        zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)),
        encoding="utf-8", errors="replace")
    games, cur = [], {}
    for line in reader:
        if line.startswith("["):
            m = re.match(r'\[(\w+) "(.*)"\]', line)
            if m:
                cur[m.group(1)] = m.group(2)
        elif line.strip():
            if "Result" in cur:
                games.append((cur.get("White"), cur.get("Black"),
                              cur.get("Result"), cur.get("ECO"),
                              cur.get("WhiteElo"), cur.get("BlackElo"),
                              cur.get("Event", "")))
            cur = {}
    df = pd.DataFrame(games, columns=["white", "black", "result", "eco",
                                      "welo", "belo", "event"])
    df.to_parquet(CACHE)
    print(f"  {len(df)} games -> {CACHE} "
          f"({os.path.getsize(CACHE)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
