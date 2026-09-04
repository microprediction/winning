"""Stream one Backblaze quarter to a compact cohort/failure table.

The quarterly zip (~1 GB, ~90 daily CSVs, one row per drive per day)
is reduced to per (date, model): number of drives observed and number
that failed that day. The SMART columns and serials are discarded;
the result is a few kilobytes, enough for the first-failure and
overdispersion tests. Model is the shared-batch factor.
"""
import collections
import csv
import io
import json
import os
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get(
    "SCRATCH", "/private/tmp/claude-501/-Users-petercotton-github-"
    "winning/4cfe1164-1ade-46c7-a29a-33bedf35fe90/scratchpad")
QUARTER = "data_Q1_2025"
URL = (f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"
       f"{QUARTER}.zip")


def main():
    zpath = os.path.join(SCRATCH, f"{QUARTER}.zip")
    if not os.path.exists(zpath):
        with requests.get(URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
    print(f"zip on disk: {os.path.getsize(zpath)/1e9:.2f} GB")

    # per (date, model): [drives observed, failures]
    table = collections.defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.endswith(".csv")]
        for mi, name in enumerate(sorted(members)):
            with z.open(name) as fh:
                rd = csv.reader(io.TextIOWrapper(fh, "utf-8"))
                header = next(rd)
                hi = {c: i for i, c in enumerate(header)}
                di, mi_, fi = (hi["date"], hi["model"], hi["failure"])
                for row in rd:
                    key = (row[di], row[mi_])
                    cell = table[key]
                    cell[0] += 1
                    if row[fi] == "1":
                        cell[1] += 1
            if (mi + 1) % 15 == 0:
                print(f"  processed {mi+1}/{len(members)} days")
    os.remove(zpath)

    out = {f"{d}|{m}": v for (d, m), v in table.items()}
    with open(os.path.join(HERE, "cohort_table.json"), "w") as f:
        json.dump(out, f)
    days = sorted(set(k.split("|")[0] for k in out))
    models = sorted(set(k.split("|")[1] for k in out))
    tot_f = sum(v[1] for v in table.values())
    print(f"{len(days)} days, {len(models)} models, "
          f"{tot_f} total failures; wrote cohort_table.json "
          f"({os.path.getsize(os.path.join(HERE,'cohort_table.json'))/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
