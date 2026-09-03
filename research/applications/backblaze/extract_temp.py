"""Second quarter + temperature: replication and factor identification.

Re-streams a Backblaze quarter keeping, per (date, model): drives,
failures, and the SUM and COUNT of SMART 194 raw (drive temperature,
Celsius). Daily fleet mean temperature = sum/count. This lets us
(a) replicate the cross-manufacturer clustering in a second quarter
and (b) test whether the shared daily failure factor is THERMAL --
whether failure spikes align with temperature.
"""
import collections
import csv
import io
import json
import os
import sys
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get(
    "SCRATCH", "/private/tmp/claude-501/-Users-petercotton-github-"
    "winning/4cfe1164-1ade-46c7-a29a-33bedf35fe90/scratchpad")
QUARTER = sys.argv[1] if len(sys.argv) > 1 else "data_Q4_2024"
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
    # per (date, model): [drives, failures, temp_sum, temp_count]
    table = collections.defaultdict(lambda: [0, 0, 0.0, 0])
    with zipfile.ZipFile(zpath) as z:
        members = sorted(m for m in z.namelist() if m.endswith(".csv"))
        for mi, name in enumerate(members):
            with z.open(name) as fh:
                rd = csv.reader(io.TextIOWrapper(fh, "utf-8-sig",
                                                 errors="replace"))
                hdr = {c.strip().lstrip("\ufeff"): i
                       for i, c in enumerate(next(rd))}
                if "date" not in hdr:
                    continue
                di, mo, fi = hdr["date"], hdr["model"], hdr["failure"]
                ti = hdr.get("smart_194_raw")
                for row in rd:
                    cell = table[(row[di], row[mo])]
                    cell[0] += 1
                    if row[fi] == "1":
                        cell[1] += 1
                    if ti is not None and row[ti]:
                        try:
                            cell[2] += float(row[ti]); cell[3] += 1
                        except ValueError:
                            pass
            if (mi + 1) % 20 == 0:
                print(f"  {mi+1}/{len(members)} days")
    os.remove(zpath)
    out = {f"{d}|{m}": v for (d, m), v in table.items()}
    fn = os.path.join(HERE, f"cohort_temp_{QUARTER}.json")
    json.dump(out, open(fn, "w"))
    tf = sum(v[1] for v in table.values())
    print(f"{QUARTER}: {len(set(k.split('|')[0] for k in out))} days, "
          f"{tf} failures; wrote {os.path.basename(fn)} "
          f"({os.path.getsize(fn)/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
