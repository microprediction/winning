"""Build the cached Lichess header table the chess experiments read.

One month of the Lichess open database (database.lichess.org, CC0),
streamed and reduced to headers only: ~380MB of PGN becomes a 1MB
parquet. Idempotent; skips the download if the cache exists.

MODERN MONTHS NEED THE CAP. 2013-01 is 380MB compressed and fits in
memory; 2024-01 is ~32GB and does not. This now streams the HTTP
response straight into the zstd decoder rather than reading it into a
bytes object first, and takes an optional cap so a modern month can be
stopped early:

    python research/chess/loader.py 2013-01
    python research/chess/loader.py 2024-01 4000000

A capped run is a contiguous PREFIX of the month, not a sample, and the
cache filename records the cap ("..._first4000000_headers.parquet") so
a truncated month can never be mistaken for a whole one.

Run:  python research/chess/loader.py [YYYY-MM] [MAX_GAMES]
"""
from __future__ import annotations
import io, os, re, sys, urllib.request
import pandas as pd, zstandard

MONTH = sys.argv[1] if len(sys.argv) > 1 else "2013-01"
MAX_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 0
_tag = f"_first{MAX_GAMES}" if MAX_GAMES else ""
CACHE = os.path.expanduser(
    f"~/.cache/winning/lichess_{MONTH.replace('-', '_')}{_tag}"
    f"_headers.parquet")
URL = (f"https://database.lichess.org/standard/"
       f"lichess_db_standard_rated_{MONTH}.pgn.zst")

def main():
    if os.path.exists(CACHE):
        print(f"cached already: {CACHE}")
        return
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"fetching {URL}"
          + (f" (stopping after {MAX_GAMES} games)" if MAX_GAMES else ""))
    resp = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "winning-research"}),
        timeout=900)
    # stream_reader over the live response: never materialises the month
    reader = io.TextIOWrapper(
        zstandard.ZstdDecompressor().stream_reader(resp),
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
                if MAX_GAMES and len(games) >= MAX_GAMES:
                    break
                if len(games) % 500_000 == 0:
                    print(f"  {len(games)} games", flush=True)
            cur = {}
    resp.close()
    df = pd.DataFrame(games, columns=["white", "black", "result", "eco",
                                      "welo", "belo", "event"])
    df.to_parquet(CACHE)
    print(f"  {len(df)} games -> {CACHE} "
          f"({os.path.getsize(CACHE)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
