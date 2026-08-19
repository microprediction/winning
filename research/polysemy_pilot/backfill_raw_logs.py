"""Seed the append-only raw logs from the older consolidated JSON caches.

One-off, idempotent: keys already in a log are skipped, so re-running adds
nothing. After this the JSONL logs hold every API response the JSON caches
held, in the format that cannot be truncated by a later partial run.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from datastore import append_jsonl, load_jsonl

PAIRS = [("perm_raw.json", "perm_raw.jsonl"),
         ("random_raw.json", "random_raw.jsonl"),
         ("exact_raw.json", "exact_raw.jsonl")]

for src, dst in PAIRS:
    if not (HERE / src).exists():
        print(f"{src}: absent, skipped")
        continue
    have = set(load_jsonl(HERE / dst, key="key"))
    added = 0
    for k, v in json.loads((HERE / src).read_text()).items():
        if k in have:
            continue
        append_jsonl(HERE / dst, {"key": k, "raw": v})
        added += 1
    print(f"{src} -> {dst}: +{added} ({len(have)} already present)")
