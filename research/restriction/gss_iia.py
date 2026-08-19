"""The decisive test: is the odds discount driven by population heterogeneity?

Ten human datasets give discount ratios from 0.78 to 3.49 against a null of one,
and the large values fall in factional domains. Simulating a mixture of Case V
groups reproduces the pattern, which suggests the ratio measures how divided a
population is rather than whether Thurstone describes an individual.

That is testable rather than merely plausible, and the General Social Survey has
what it takes. Its socialization-values item is a rank-order card task: pick the
most important thing for a child to learn, then the next, and so on, over obey,
popular, think for self, work hard, help others. Thirty-four thousand respondents
supply exact permutations with no ties by construction. The obey-versus-think-for-
self contrast is the classic authoritarianism axis, so the population is divided
in a way that is measured by other variables in the same file.

The test: compute the ratio for the whole sample, then within subgroups that share
education, politics and religious attendance. If heterogeneity produces the ratio,
it should fall toward one inside a homogeneous subgroup while remaining high in the
pooled sample. If it stays high everywhere, the mixture explanation fails and the
departure belongs to individuals.

Data: https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip
(free, no registration; 47 MB zipped, 598 MB as Stata).

Usage:  python gss_iia.py path/to/gss7224_r3a.dta
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

SOCIAL = ["obey", "popular", "thnkself", "workhard", "helpoth"]
JOB = ["jobinc", "jobsec", "jobhour", "jobpromo", "jobmeans"]
COVARS = ["year", "degree", "polviews", "attend", "age"]


def shares(R, cols):
    sub = R[:, cols]
    best = sub.min(axis=1, keepdims=True)
    top = sub == best
    n = top.sum(axis=1)
    w = top / n[:, None]
    tot = w.sum()
    return (w.sum(axis=0) / tot, len(sub)) if tot > 0 else (None, 0)


def pairs_for(R):
    K = R.shape[1]
    full, n = shares(R, list(range(K)))
    if full is None or n < 200:
        return None
    live = [i for i in range(K) if full[i] > 0]
    p = [full[i] for i in live]
    z = sum(p)
    a_loc, err = calibrate_np([x / z for x in p])
    if err > 0.05:
        return None
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pr, m = shares(R, [i, j])
            if pr is None or pr[0] <= 0 or pr[1] <= 0:
                continue
            L = math.log(full[i] / full[j])
            if abs(L) < 1e-6:
                continue
            w = win_probs_np(a_loc[[x, y]])
            if w[0] <= 0 or w[1] <= 0:
                continue
            obs.append((L, math.log(pr[0] / pr[1]) - L))
            cv.append((L, math.log(w[0] / w[1]) - L))
    return (obs, cv, n) if len(obs) >= 3 else None


def lam(P):
    return sum(-d * L for L, d in P) / sum(L * L for L, _ in P)


def ratio_ci(obs, cv, B=8000, seed=11):
    random.seed(seed)
    idx, out = list(range(len(obs))), []
    for _ in range(B):
        s = [idx[random.randrange(len(idx))] for _ in idx]
        no = sum(-obs[k][1] * obs[k][0] for k in s)
        do = sum(obs[k][0] ** 2 for k in s)
        nc = sum(-cv[k][1] * cv[k][0] for k in s)
        dc = sum(cv[k][0] ** 2 for k in s)
        if do > 0 and dc > 0 and nc != 0:
            out.append((no / do) / (nc / dc))
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def report(label, R):
    r = pairs_for(R)
    if not r:
        print(f"  {label:<44} too few complete rankings")
        return None
    obs, cv, n = r
    rt = lam(obs) / lam(cv)
    lo, hi = ratio_ci(obs, cv)
    print(f"  {label:<44} n={n:>6}  lambda {lam(obs):.3f}  "
          f"ratio {rt:.2f} [{lo:.2f}, {hi:.2f}]")
    return rt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gss7224_r3a.dta"
    d = pd.read_stata(path, columns=SOCIAL + JOB + COVARS,
                      convert_categoricals=False)
    print(f"GSS cumulative file: {len(d):,} respondents\n")

    for name, cols in (("socialization values", SOCIAL), ("job values", JOB)):
        M = d[cols].to_numpy(dtype=float)
        ok = np.array([len(set(r)) == len(cols) and not np.isnan(r).any()
                       and set(r) == set(range(1, len(cols) + 1))
                       for r in M])
        R = M[ok]
        sub = d[ok]
        print(f"{name}: {len(R):,} complete strict rankings")
        report("whole sample", R)

        # within-subgroup: if heterogeneity drives the ratio it should fall here
        print("  within homogeneous subgroups:")
        for cv_name, label in (("degree", "education"),
                               ("polviews", "political views"),
                               ("attend", "religious attendance")):
            if cv_name not in sub:
                continue
            vals = sub[cv_name].dropna().unique()
            for v in sorted(vals)[:7]:
                sel = (sub[cv_name] == v).to_numpy()
                if sel.sum() < 1500:
                    continue
                report(f"    {label} = {v:g}", R[sel])
        print()

    print("null under a homogeneous Gaussian contest: 1.00")
    print("simulated two-faction mixture: about 4; three factions about 2.5")
    print("machines: 5.77 [4.32, 7.65]")


if __name__ == "__main__":
    main()
