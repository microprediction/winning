"""Line-length choice with induced values, where linear renormalization wins.

Duffy and Smith (2025) show subjects two to six grey lines and pay them for selecting the
longest. Value is induced and one-dimensional by construction, which the authors make the
selling point of the design. That places the alternatives on a single physical continuum,
which is where this paper's boundary rule says linear renormalization should be the better
map, so the dataset is a prediction rather than a threat.

The paper's usual protocol does not apply here. Lengths are redrawn each trial, so there are
no fixed alternatives carrying stable shares to calibrate from, and no share vector to invert.
What can be done instead is the parametric version of the same question: fit one scale
parameter per model on menus of one size, then score menus of another size out of sample.

  Luce with iid Gumbel      P(i) proportional to exp(len_i / beta)
  Thurstone Case V          P(i) = P(len_i + sigma Z_i is largest), Z iid standard normal

One free parameter each, fitted by maximum likelihood on the calibration stratum only, then
held fixed while the other stratum is scored by log likelihood per observation. The middle
row of the table is the restriction direction proper: calibrate on the six-line menus and
predict the two-line menus.

This is a fitted comparison, not the parameter-free one the rest of the paper runs. It is
reported because it is the same question asked on this dataset, and because a referee who
runs it will get this answer.

Usage:  python duffy_smith.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "duffy_smith" / "NoCLUALData-OSF.csv"
GRID = np.linspace(-9, 9, 721)          # for the Case V integral, in units of sigma


def load():
    """Trials as (lengths, chosen index), dropping rows without a usable selection."""
    trials = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            try:
                n = int(float(r["NumLines"]))
                sel = int(float(r["LineSelected"]))
            except (ValueError, KeyError):
                continue
            if not (2 <= n <= 6) or not (0 <= sel < n):
                continue
            lens = []
            ok = True
            for k in range(n):
                v = r.get(f"LineLength{k}", "")
                try:
                    lens.append(float(v))
                except ValueError:
                    ok = False
                    break
            if not ok or len(lens) != n:
                continue
            trials.append((np.array(lens), sel, n))
    return trials


def luce_ll(trials, beta):
    """Log likelihood per observation under Luce with scale beta."""
    tot = 0.0
    for lens, sel, _ in trials:
        z = (lens - lens.max()) / beta
        e = np.exp(z)
        tot += z[sel] - np.log(e.sum())
    return tot / len(trials)


def casev_probs(lens, sigma):
    """P(i wins) for locations lens/sigma against iid standard normal noise."""
    a = lens / sigma
    out = np.empty(len(a))
    for i in range(len(a)):
        x = GRID + a[i]
        dens = norm.pdf(x - a[i])
        for k in range(len(a)):
            if k != i:
                dens = dens * norm.cdf(x - a[k])
        out[i] = np.trapezoid(dens, x)
    s = out.sum()
    return out / s if s > 0 else np.full(len(a), 1.0 / len(a))


def casev_ll(trials, sigma):
    tot = 0.0
    for lens, sel, _ in trials:
        p = casev_probs(lens, sigma)
        tot += np.log(max(p[sel], 1e-12))
    return tot / len(trials)


def fit(trials, ll, lo, hi):
    r = minimize_scalar(lambda t: -ll(trials, t), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-3})
    return float(r.x), float(-r.fun)


def main():
    trials = load()
    print(f"{len(trials)} valid trials from {DATA.name}")
    by_n = {}
    for t in trials:
        by_n.setdefault(t[2], []).append(t)
    print("  by menu size: " + ", ".join(f"n={k}: {len(v)}" for k, v in sorted(by_n.items())))

    splits = [
        ("n=2", [2], "n=3..6", [3, 4, 5, 6]),
        ("n=6", [6], "n=2", [2]),
        ("n=5,6", [5, 6], "n=2,3", [2, 3]),
    ]
    print(f"\n{'calibrate':<10}{'predict':<10}{'N cal':>7}{'N pred':>8}"
          f"{'beta':>9}{'sigma px':>10}{'Luce LL':>10}{'CaseV LL':>10}"
          f"{'advantage':>11}{'SE':>9}{'t':>7}")
    for cal_name, cal_ns, pred_name, pred_ns in splits:
        cal = [t for n in cal_ns for t in by_n.get(n, [])]
        pred = [t for n in pred_ns for t in by_n.get(n, [])]
        if not cal or not pred:
            continue
        beta, _ = fit(cal, luce_ll, 0.01, 200.0)
        sigma, _ = fit(cal, casev_ll, 0.5, 200.0)
        # per-observation log likelihoods on the held-out stratum
        dl = []
        for lens, sel, _ in pred:
            z = (lens - lens.max()) / beta
            l_luce = z[sel] - np.log(np.exp(z).sum())
            l_case = np.log(max(casev_probs(lens, sigma)[sel], 1e-12))
            dl.append(l_luce - l_case)
        dl = np.array(dl)
        adv, se = dl.mean(), dl.std(ddof=1) / np.sqrt(len(dl))
        print(f"{cal_name:<10}{pred_name:<10}{len(cal):>7}{len(pred):>8}"
              f"{beta:>9.4f}{sigma:>10.2f}{luce_ll(pred, beta):>10.4f}"
              f"{casev_ll(pred, sigma):>10.4f}{adv:>+11.4f}{se:>9.4f}{adv / se:>7.2f}")
    print("\nA positive advantage favours Luce, that is linear renormalization. The second row")
    print("is the restriction direction: calibrate on the full menu, predict the reduced one.")


if __name__ == "__main__":
    main()
