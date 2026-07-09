"""Control experiment: the retirement block should do nothing for qualifying.

Qualifying is a pace contest with almost no retirement process, so if the
disaster block's value on races comes from modeling retirement rather than
from generic tail mass, it should not help (and may hurt) when the same
systems rate QUALIFYING sessions as contests in their own right. Same
protocol, same density candidates, target = qualifying classification.

Run:  .venv/bin/python research/f1_quali_control.py

Measured (July 2026, 1,158 sessions):
    gaussian            1.8498 / tau 0.609 / pit 0.025
    fixed block p=0.25  1.8430 / tau 0.564 / pit 0.088   (damages order+shape)
    era-adaptive block  1.8419 / tau 0.605 / pit 0.023   final p = 0.03
The adaptive estimator drives the block mass to ~0.03 where the retirement
mechanism is absent, restoring Gaussian-level tau and the best calibration;
forcing the 25% block onto qualifying damages rank correlation and PIT.
The block models retirement, not tails in general. Side-reading: quali tau
0.61 vs race 0.33 — qualifying is the predictable half of the weekend.
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import zipfile
from collections import defaultdict

from winning import ThurstoneRating
from winning.benchmarks.events import Event
from winning.benchmarks.forward_chain import evaluate
from f1_dirac_disaster import dirac_disaster_kernel
from f1_era_slab import EraSlabThurstoneRating


def quali_events():
    path = os.path.expanduser("~/.cache/winning/f1db-csv.zip")
    with zipfile.ZipFile(path) as zf:
        rd = {}
        with zf.open("f1db-races.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rd[row["id"]] = row["date"]
        by_race = defaultdict(list)
        with zf.open("f1db-races-qualifying-results.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                by_race[row["raceId"]].append(row)

    dated = []
    for race_id, rows in by_race.items():
        date = rd.get(race_id)
        if not date:
            continue
        ordinal = datetime.date.fromisoformat(date).toordinal()
        names, ranks = [], []
        worst = 0
        finishers, dns = [], []
        seen = set()
        for row in sorted(rows, key=lambda r: int(r.get("positionDisplayOrder") or 0)):
            drv = row["driverId"]
            if drv in seen:
                continue
            seen.add(drv)
            pos = row.get("positionNumber")
            if pos:
                finishers.append((int(pos), drv))
                worst = max(worst, int(pos))
            else:
                dns.append(drv)
        if len(finishers) + len(dns) < 2:
            continue
        for pos, drv in sorted(finishers):
            names.append(drv)
            ranks.append(pos)
        for drv in dns:
            names.append(drv)
            ranks.append(worst + 1)
        dated.append((ordinal, Event(names=names, ranks=ranks)))
    dated.sort(key=lambda t: t[0])
    events = []
    prev = None
    for ordinal, ev in dated:
        ev.dt = 0.0 if prev is None else float(ordinal - prev)
        prev = ordinal
        events.append(ev)
    return events


def main():
    events = quali_events()
    print(f"qualifying sessions as contests: {len(events)}\n")
    for label, system in [
        ("gaussian", ThurstoneRating()),
        ("fixed block p=0.25", ThurstoneRating(
            base_kernel=dirac_disaster_kernel(core_sd=1.0, p_disaster=0.25))),
        ("era-adaptive block", EraSlabThurstoneRating()),
    ]:
        s = evaluate(system, events, rank_pit=True).summary()
        extra = f"  (final p={system.current_p:.2f})" if hasattr(system, "current_p") else ""
        print(f"{label:20s} log_loss={s['log_loss']:.4f} acc={s['accuracy']:.4f} "
              f"tau={s['kendall_tau']:.4f} pit_ks={s['rank_pit_ks']:.4f}{extra}")


if __name__ == "__main__":
    main()
