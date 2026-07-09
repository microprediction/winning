"""Enumerate and download Wayback-archived Oddschecker F1 winner pages.

Three URL eras (2007-2010, 2013-2017, 2019-2025); CDX enumeration, then
throttled downloads of one snapshot per (race-page, calendar-proximity)
into ~/.cache/winning/oddschecker/. Personal research use of archived
pages; only derived consensus probabilities are ever published.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

CDX = "https://web.archive.org/cdx/search/cdx"
PATTERNS = [
    "oddschecker.com/motor-sport/formula-one/*",
    "oddschecker.com/motorsport/formula-one/*",
    "oddschecker.com/motorsport/formula-1/*",
]
OUT = os.path.expanduser("~/.cache/winning/oddschecker")


def cdx_rows(pattern):
    q = urllib.parse.urlencode({
        "url": pattern, "output": "json", "collapse": "timestamp:8",
        "filter": "statuscode:200", "limit": "5000",
    })
    req = urllib.request.Request(f"{CDX}?{q}", headers={"User-Agent": "winning-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = json.load(r)
    return rows[1:] if rows else []


def main():
    os.makedirs(OUT, exist_ok=True)
    keep = []
    for pat in PATTERNS:
        rows = cdx_rows(pat)
        for ts, orig in [(r[1], r[2]) for r in rows]:
            low = orig.lower()
            if ("winner" in low or "win-market" in low) and "each-way" not in low \
                    and "outright" not in low and "championship" not in low:
                keep.append((ts, orig))
        time.sleep(2)
    keep.sort()
    print(f"candidate snapshots: {len(keep)}")
    manifest = os.path.join(OUT, "manifest.tsv")
    with open(manifest, "w") as f:
        for ts, orig in keep:
            f.write(f"{ts}\t{orig}\n")

    got, errs = 0, 0
    for i, (ts, orig) in enumerate(keep):
        fn = os.path.join(OUT, f"{ts}_{urllib.parse.quote(orig, safe='')[:180]}.html")
        if os.path.exists(fn):
            got += 1
            continue
        url = f"https://web.archive.org/web/{ts}id_/{orig}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "winning-research"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(fn, "wb") as f:
                f.write(data)
            got += 1
        except Exception as e:
            errs += 1
        time.sleep(1.5)  # be polite to archive.org
        if i % 25 == 0:
            print(f"{i}/{len(keep)} fetched={got} errors={errs}", flush=True)
    print(f"done: fetched={got} errors={errs}")


if __name__ == "__main__":
    main()
