"""Every usable Oddschecker winner snapshot per race (the repo's parser
keeps only the latest), as rows race_key, snapshot_ts, driver, n_bookies,
consensus_prob. Reuses the parser's classify / read_page / slug_fraction /
parse_rows and thresholds."""
import csv, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # research/f1_odds_parse.py
import f1_odds_parse as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_market_all.csv")


def consensus(drivers):
    import numpy as np
    inv = {d: float(np.median([1.0 / o for o in odds])) for d, odds in drivers.items() if len(odds) >= P.MIN_BOOKIES}
    tot = sum(inv.values())
    return {d: (len(drivers[d]), v / tot) for d, v in inv.items()}


def main():
    by_race = defaultdict(list)
    for fname in os.listdir(P.CACHE_DIR):
        info = P.classify(fname)
        if info is None:
            continue
        ts, race = info
        by_race[f"{ts[:4]}_{race}"].append((ts, fname, race))
    rows, n_snap = [], 0
    for race_key in sorted(by_race):
        for ts, fname, slug in sorted(by_race[race_key]):
            page = P.read_page(os.path.join(P.CACHE_DIR, fname))
            if P.slug_fraction(page, slug) < 0.5:
                continue
            drivers = P.parse_rows(page)
            good = {d: o for d, o in drivers.items() if len(o) >= P.MIN_BOOKIES}
            if len(good) < P.MIN_DRIVERS:
                continue
            n_snap += 1
            for d, (nb, p) in consensus(drivers).items():
                rows.append((race_key, ts, d, nb, f"{p:.6f}"))
    # never silently shrink the tracked file: a fetch that missed pages would
    # otherwise remove market evidence from later season runs
    if os.path.exists(OUT) and "--allow-fewer" not in sys.argv:
        have = len({(r["race_key"], r["snapshot_ts"]) for r in csv.DictReader(open(OUT))})
        if n_snap < have:
            raise SystemExit(f"refusing to replace {OUT}: {n_snap} snapshots parsed, {have} already there; "
                             "pass --allow-fewer to accept the smaller set")
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["race_key", "snapshot_ts", "driver", "n_bookies", "consensus_prob"]); w.writerows(rows)
    os.replace(tmp, OUT)                       # the file is complete or unchanged, never partial
    print(f"{n_snap} snapshots over {len({r[0] for r in rows})} races -> {OUT}")


if __name__ == "__main__":
    main()
