"""Replicate the common-cause finding in a second quarter, and test
whether the shared factor is thermal.

Input: cohort_temp_<QUARTER>.json with per (date, model) drives,
failures, temp_sum, temp_count.

  1. Replication: fleet dispersion (raw and detrended), spike days
     and how many models they span, cross-manufacturer detrended
     correlation -- the same battery as analyze2.py, on new data.
  2. Thermal identification: daily fleet mean drive temperature vs
     the detrended daily failure residual; whether spike days are hot;
     and whether partialling temperature out of the manufacturers'
     residuals REDUCES their cross-correlation (temperature as at
     least part of the shared factor).
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QUARTER = sys.argv[1] if len(sys.argv) > 1 else "data_Q4_2024"


def manufacturer(model):
    m = model.upper()
    if m.startswith("ST"):
        return "Seagate"
    for name in ("TOSHIBA", "HGST", "WDC", "SEAGATE"):
        if m.startswith(name):
            return name.title()
    return "Other"


def smooth(x, w=7):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, w // 2, mode="edge"),
                       np.ones(w) / w, "valid")


def detrend_counts(coh, fail):
    hz = fail / np.maximum(coh, 1)
    exp = coh * smooth(hz, 7)
    return fail - exp, exp


if __name__ == "__main__":
    d = json.load(open(os.path.join(HERE,
                                    f"cohort_temp_{QUARTER}.json")))
    pm = collections.defaultdict(dict)
    for k, v in d.items():
        day, model = k.split("|", 1)
        pm[model][day] = v
    days = sorted({day for e in pm.values() for day in e})
    T = len(days)

    def series(entries):
        coh = np.array([entries.get(x, [0, 0, 0, 0])[0] for x in days],
                       float)
        fail = np.array([entries.get(x, [0, 0, 0, 0])[1] for x in days],
                        float)
        tsum = np.array([entries.get(x, [0, 0, 0, 0])[2] for x in days],
                        float)
        tcnt = np.array([entries.get(x, [0, 0, 0, 0])[3] for x in days],
                        float)
        return coh, fail, tsum, tcnt

    fleet_c = np.zeros(T); fleet_f = np.zeros(T)
    fleet_ts = np.zeros(T); fleet_tc = np.zeros(T)
    for e in pm.values():
        c, f, ts, tc = series(e)
        fleet_c += c; fleet_f += f; fleet_ts += ts; fleet_tc += tc
    temp = fleet_ts / np.maximum(fleet_tc, 1)          # daily mean C

    raw = fleet_f.var() / fleet_f.mean()
    resid, exp = detrend_counts(fleet_c, fleet_f)
    det = float(np.var(resid) / np.mean(exp))
    print(f"[{QUARTER}] fleet daily failures: raw Var/Mean {raw:.2f} "
          f"-> detrended {det:.2f}; mean drive temp "
          f"{temp.mean():.1f}C (range {temp.min():.1f}-{temp.max():.1f})")

    std = resid / np.sqrt(np.maximum(exp, 1e-9))
    spikes = [i for i in range(T) if std[i] > 3]
    temp_z = (temp - temp.mean()) / temp.std()
    print(f"spike days (>3 sigma): {len(spikes)}; mean temp-z on "
          f"spike days {np.mean([temp_z[i] for i in spikes]):+.2f} "
          f"(vs 0 fleet-wide)" if spikes else "no spike days")

    # cross-manufacturer detrended residuals
    man_c = collections.defaultdict(lambda: np.zeros(T))
    man_f = collections.defaultdict(lambda: np.zeros(T))
    for model, e in pm.items():
        c, f, _, _ = series(e)
        man_c[manufacturer(model)] += c
        man_f[manufacturer(model)] += f
    big = sorted(man_f, key=lambda k: -man_f[k].sum())[:4]
    R = {k: detrend_counts(man_c[k], man_f[k])[0] for k in big}

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def partial(a, b, z):
        # correlation of a,b residualized on z
        ra = a - np.polyval(np.polyfit(z, a, 1), z)
        rb = b - np.polyval(np.polyfit(z, b, 1), z)
        return corr(ra, rb)

    print("cross-manufacturer detrended correlation, and after "
          "partialling out temperature:")
    raw_offs, par_offs = [], []
    for i, x in enumerate(big):
        for y in big[i + 1:]:
            c0 = corr(R[x], R[y]); cp = partial(R[x], R[y], temp)
            raw_offs.append(c0); par_offs.append(cp)
            print(f"  {x:8s} x {y:8s}: {c0:+.3f} -> {cp:+.3f} "
                  f"(temp partialled)")
    tf_corr = corr(resid, temp)
    print(f"  mean off-diagonal: {np.mean(raw_offs):+.3f} -> "
          f"{np.mean(par_offs):+.3f} after temperature")
    print(f"fleet failure-residual vs temperature correlation: "
          f"{tf_corr:+.3f}")

    json.dump(dict(quarter=QUARTER, raw_disp=float(raw),
                   detrended_disp=det, n_spikes=len(spikes),
                   mean_cross_corr=float(np.mean(raw_offs)),
                   mean_cross_corr_temp_partialled=float(
                       np.mean(par_offs)),
                   failure_temp_corr=tf_corr,
                   spike_temp_z=[float(temp_z[i]) for i in spikes]),
              open(os.path.join(HERE,
                                f"results_temp_{QUARTER}.json"), "w"),
              indent=2)
    print("wrote results_temp.json")
