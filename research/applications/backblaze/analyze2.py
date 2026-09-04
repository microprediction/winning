"""Does the Backblaze common-cause signal survive aging control?

The referee objection to analyze.py: a constant-hazard null makes
drift (aging, cohort turnover) look like overdispersion. The fix is
to separate the two by their signature. Aging is a SLOW per-model
trend and cannot make different drives fail the SAME calendar day.
Genuine durability-relevant common cause (power, cooling, humidity,
a handling event) makes MANY drives fail together on a day, ACROSS
models. So:

  1. Cohort-adjust and detrend the fleet daily failure count; does
     the overdispersion survive (same-day clustering that a smooth
     trend cannot explain)?
  2. Spike days: days whose failure count exceeds the Poisson
     expectation by many sigma. Are they concentrated in one model
     (a batch event) or spread across models (an environmental
     event)?
  3. The killer test: after removing each MANUFACTURER's own smooth
     trend, are the manufacturers' daily failure residuals
     POSITIVELY CORRELATED with each other? Seagate and HGST fail by
     unrelated mechanisms; same-day co-movement of their detrended
     failures can only be a shared environmental factor -- the
     common cause aging cannot fake, and exactly the factor the
     durability model needs.
"""
import collections
import json
import os

import numpy as np
from scipy.stats import poisson

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = json.load(open(os.path.join(HERE, "cohort_table.json")))
    per_model = collections.defaultdict(dict)
    for k, (n, f) in d.items():
        day, model = k.split("|", 1)
        per_model[model][day] = (n, f)
    return per_model


def manufacturer(model):
    m = model.upper()
    for name in ("TOSHIBA", "HGST", "WDC", "SEAGATE", "ST", "DELLBOSS",
                 "MTFDDAV", "MICRON"):
        if m.startswith(name):
            return "Seagate" if name == "ST" else name.title()
    return "Other"


def smooth(x, w=7):
    """Centered moving-average trend (odd window)."""
    x = np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.pad(x, w // 2, mode="edge"), k, "valid")


def daily_series(entries, days):
    coh = np.array([entries.get(d, (0, 0))[0] for d in days], float)
    fail = np.array([entries.get(d, (0, 0))[1] for d in days], float)
    return coh, fail


if __name__ == "__main__":
    pm = load()
    days = sorted({d for e in pm.values() for d in e})
    T = len(days)

    # fleet series
    fleet_coh = np.zeros(T)
    fleet_fail = np.zeros(T)
    for e in pm.values():
        c, f = daily_series(e, days)
        fleet_coh += c
        fleet_fail += f

    raw_disp = fleet_fail.var() / fleet_fail.mean()
    # cohort-adjusted, detrended expectation: smooth per-drive hazard
    hazard = fleet_fail / np.maximum(fleet_coh, 1)
    exp_fail = fleet_coh * smooth(hazard, 7)
    # Pearson residuals vs the detrended Poisson mean
    resid = (fleet_fail - exp_fail) / np.sqrt(np.maximum(exp_fail, 1e-9))
    detrended_disp = float(np.var(fleet_fail - exp_fail)
                           / np.mean(np.maximum(exp_fail, 1e-9)))
    print(f"fleet daily failures: raw Var/Mean {raw_disp:.2f}  ->  "
          f"cohort-adjusted+detrended dispersion {detrended_disp:.2f} "
          f"(1.0 = Poisson; >1 survives aging control)")

    # spike days: standardized residual > 3
    spikes = [i for i in range(T) if resid[i] > 3.0]
    print(f"\nspike days (detrended residual > 3 sigma): {len(spikes)}")
    for i in spikes[:8]:
        # which models contributed the excess that day
        contrib = sorted(
            ((f, m) for m, e in pm.items()
             for (dd, (nn, f)) in [(days[i], e.get(days[i], (0, 0)))]
             if f > 0), reverse=True)[:4]
        nmodels = sum(1 for m, e in pm.items()
                      if e.get(days[i], (0, 0))[1] > 0)
        tot = int(fleet_fail[i])
        print(f"  {days[i]}: {tot} failures (exp {exp_fail[i]:.0f}), "
              f"{nmodels} distinct models, top "
              + ", ".join(f"{m.split()[0]}:{f}" for f, m in contrib))

    # manufacturer detrended residual cross-correlation (killer test)
    man = collections.defaultdict(lambda: np.zeros(T))
    man_coh = collections.defaultdict(lambda: np.zeros(T))
    for model, e in pm.items():
        c, f = daily_series(e, days)
        man[manufacturer(model)] += f
        man_coh[manufacturer(model)] += c
    big = sorted(man, key=lambda k: -man[k].sum())[:4]
    resids = {}
    for k in big:
        h = man[k] / np.maximum(man_coh[k], 1)
        ex = man_coh[k] * smooth(h, 7)
        resids[k] = man[k] - ex
    print(f"\nmanufacturer detrended-residual cross-correlation "
          f"(same-day co-movement; >0 = shared environmental factor):")
    offs = []
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            r = np.corrcoef(resids[a], resids[b])[0, 1]
            offs.append(r)
            print(f"  {a:9s} x {b:9s}: {r:+.3f}  "
                  f"({int(man[a].sum())} vs {int(man[b].sum())} fails)")
    print(f"  mean off-diagonal correlation: {np.mean(offs):+.3f}")

    json.dump(dict(raw_dispersion=float(raw_disp),
                   detrended_dispersion=detrended_disp,
                   n_spike_days=len(spikes),
                   spike_days=[days[i] for i in spikes],
                   mean_cross_manufacturer_corr=float(np.mean(offs))),
              open(os.path.join(HERE, "results_detrended.json"), "w"),
              indent=2)
    print("\nwrote results_detrended.json")
