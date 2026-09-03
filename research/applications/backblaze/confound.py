"""Is the cross-manufacturer clustering physical, or a reporting
artifact? The skeptic's tests, on both quarters.

If Backblaze records failures on an administrative cadence (drives
pulled/marked in weekday batches, not on the physical failure day),
then all manufacturers spike on the same ADMIN days and the
cross-manufacturer correlation is an artifact, not common cause. Tests:
  1. Day-of-week profile of failures. Strong weekday>weekend
     structure is the signature of batch processing.
  2. Are the >3-sigma spike days concentrated on particular weekdays?
  3. THE decisive one: does the cross-manufacturer correlation
     SURVIVE removing day-of-week means? If partialling out
     day-of-week collapses it, the "common cause" was the maintenance
     calendar. If it survives, the clustering is not merely the
     admin cadence.
"""
import collections
import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def manufacturer(model):
    m = model.upper()
    if m.startswith("ST"):
        return "Seagate"
    for name in ("TOSHIBA", "HGST", "WDC"):
        if m.startswith(name):
            return name.title()
    return "Other"


def smooth(x, w=7):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, w // 2, mode="edge"),
                       np.ones(w) / w, "valid")


def load(fname):
    d = json.load(open(os.path.join(HERE, fname)))
    pm = collections.defaultdict(dict)
    for k, v in d.items():
        day, model = k.split("|", 1)
        pm[model][day] = v[:2]
    return pm


def analyze(fname, label):
    pm = load(fname)
    days = sorted({x for e in pm.values() for x in e})
    wd = np.array([dt.date.fromisoformat(x).weekday() for x in days])

    def series(entries):
        return np.array([entries.get(x, [0, 0])[1] for x in days],
                        float)

    fleet = np.zeros(len(days))
    for e in pm.values():
        fleet += series(e)

    # 1. day-of-week profile
    prof = np.array([fleet[wd == k].mean() for k in range(7)])
    print(f"[{label}] failures/day by weekday: "
          + "  ".join(f"{DOW[k]} {prof[k]:.1f}" for k in range(7)))
    weekday = fleet[wd < 5].mean()
    weekend = fleet[wd >= 5].mean()
    print(f"  weekday {weekday:.1f} vs weekend {weekend:.1f} "
          f"(ratio {weekday/max(weekend,1e-9):.2f})")

    # 2. spike-day weekdays
    hz = fleet / np.maximum(sum(np.array([e.get(x,[0,0])[0]
                                          for x in days], float)
                                for e in pm.values()), 1)
    coh = sum(np.array([e.get(x, [0, 0])[0] for x in days], float)
              for e in pm.values())
    exp = coh * smooth(fleet / np.maximum(coh, 1), 7)
    std = (fleet - exp) / np.sqrt(np.maximum(exp, 1e-9))
    spikes = np.where(std > 3)[0]
    print(f"  spike days on: "
          + ", ".join(f"{days[i]}({DOW[wd[i]]})" for i in spikes))

    # 3. cross-manufacturer correlation, raw vs day-of-week-removed
    man_f = collections.defaultdict(lambda: np.zeros(len(days)))
    man_c = collections.defaultdict(lambda: np.zeros(len(days)))
    for model, e in pm.items():
        man_f[manufacturer(model)] += series(e)
        man_c[manufacturer(model)] += np.array(
            [e.get(x, [0, 0])[0] for x in days], float)
    big = sorted(man_f, key=lambda k: -man_f[k].sum())[:4]

    def resid(f, c):
        return f - c * smooth(f / np.maximum(c, 1), 7)

    def dow_remove(x):
        y = x.copy()
        for k in range(7):
            m = wd == k
            y[m] = x[m] - x[m].mean()
        return y

    raw, ctrl = [], []
    for i, a in enumerate(big):
        ra, rca = resid(man_f[a], man_c[a]), None
        for b in big[i + 1:]:
            rb = resid(man_f[b], man_c[b])
            c0 = np.corrcoef(ra, rb)[0, 1]
            c1 = np.corrcoef(dow_remove(ra), dow_remove(rb))[0, 1]
            raw.append(c0); ctrl.append(c1)
    print(f"  mean cross-manufacturer corr: {np.mean(raw):+.3f} "
          f"raw -> {np.mean(ctrl):+.3f} after removing day-of-week")
    return dict(label=label, dow_profile=prof.tolist(),
                weekday=float(weekday), weekend=float(weekend),
                spike_dows=[int(wd[i]) for i in spikes],
                cross_raw=float(np.mean(raw)),
                cross_dow_removed=float(np.mean(ctrl)))


if __name__ == "__main__":
    out = {}
    out["Q1_2025"] = analyze("cohort_table.json", "Q1 2025")
    out["Q4_2024"] = analyze("cohort_temp_data_Q4_2024.json", "Q4 2024")
    json.dump(out, open(os.path.join(HERE, "results_confound.json"),
                        "w"), indent=2)
    print("wrote results_confound.json")
