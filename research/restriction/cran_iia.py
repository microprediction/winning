"""Human odds invariance across further constructs, from R package data.

Extends the human panel beyond sushi, Netflix, Jester and the perceptual sets.
Everything here is a complete per-respondent ranking of between three and twelve
alternatives from at least a hundred people, verified rather than taken from
documentation.

Two storage conventions coexist and confusing them silently transposes the data.
BayesMallows, ConsRank and PerMallows store RANKS, so element j is the rank given
to alternative j and the favourite is the argmin. PLMIX stores ORDERINGS, so the
first element names the favourite. The German political-goals data ships in both
ConsRank and PLMIX and the two agree exactly once each is read under its own
convention, which is how the conventions were established here; sushi agrees with
the PrefLib file under the rank convention.

For each dataset we compute the discount lambda in delta = -lambda log(p_i/p_j)
and its ratio to the discount a Case V contest predicts for the same
distribution. That ratio has an exact null of one under any Gaussian contest.

Requires the CRAN source tarballs; see FILES for the download paths.

Usage:  python cran_iia.py /path/to/extracted/cran
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

# (package, file, object, convention, construct, description)
FILES = [
    ("ConsRank",   "APAred.rda",       "APAred",      "rank",  "election",
     "APA 1980 presidential ballot, 5 candidates"),
    ("PLMIX",      "d_dublinwest.RData","d_dublinwest","order", "election",
     "Dublin West 2002 general election, 9 candidates"),
    ("PLMIX",      "d_rice.RData",     "d_rice",      "order", "election",
     "Rice University search committee, 5 candidates"),
    ("ConsRank",   "German.rda",       "German",      "rank",  "political values",
     "desirability of 4 political goals, Political Action survey"),
    ("PLMIX",      "d_occup.RData",    "d_occup",     "order", "social judgment",
     "perceived prestige of 10 occupations"),
    ("ConsRank",   "sports.rda",       "sports",      "rank",  "leisure",
     "preference for participating in 7 sports"),
    ("PLMIX",      "d_carconf.RData",  "d_carconf",   "order", "consumer",
     "importance of 6 car-configurator attributes"),
]


def load(root, pkg, fn, obj, conv):
    p = Path(root) / pkg / "data" / fn
    if not p.exists():
        return None
    d = pyreadr.read_r(str(p))
    key = obj if obj in d else list(d)[0]
    M = np.asarray(d[key], dtype=float)
    M = M[~np.isnan(M).any(axis=1)]
    K = M.shape[1]
    if conv == "order":
        # orderings pad incomplete ballots with zeros; keep complete rows only
        M = M[(M >= 1).all(axis=1)]
        rank = np.empty_like(M)
        for r in range(M.shape[0]):
            for pos, item in enumerate(M[r].astype(int)):
                rank[r, item - 1] = pos + 1
        M = rank
    else:
        M = M[(M >= 1).all(axis=1)]
    # keep only exact permutations, so nothing is imputed
    good = np.array([len(set(row.astype(int))) == K for row in M])
    return M[good]


def shares(R, cols):
    sub = R[:, cols]
    best = sub.min(axis=1, keepdims=True)
    top = sub == best
    n = top.sum(axis=1)
    w = top / n[:, None]
    tot = w.sum()
    return (w.sum(axis=0) / tot, len(sub)) if tot > 0 else (None, 0)


def lam(pairs, seed=4, B=8000):
    num = sum(-d * L for L, d in pairs)
    den = sum(L * L for L, _ in pairs)
    est = num / den
    random.seed(seed)
    idx, bs = list(range(len(pairs))), []
    for _ in range(B):
        s = [pairs[random.randrange(len(idx))] for _ in idx]
        de = sum(L * L for L, _ in s)
        if de > 0:
            bs.append(sum(-d * L for L, d in s) / de)
    bs.sort()
    return est, bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]


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


def analyse(R):
    K = R.shape[1]
    full, n = shares(R, list(range(K)))
    if full is None or n < 100:
        return None
    live = [i for i in range(K) if full[i] > 0]
    if len(live) < 3:
        return None
    p = [full[i] for i in live]
    z = sum(p)
    p = [x / z for x in p]
    a_loc, err = calibrate_np(p)
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


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"{'construct':<17}{'dataset':<14}{'n':>7}{'pairs':>7}"
          f"{'lambda':>8}{'ratio to Case V':>22}")
    pooled_o, pooled_c = [], []
    for pkg, fn, obj, conv, con, desc in FILES:
        R = load(root, pkg, fn, obj, conv)
        if R is None or len(R) < 100:
            print(f"{con:<17}{obj:<14}  unavailable or too few complete rankings")
            continue
        r = analyse(R)
        if not r:
            print(f"{con:<17}{obj:<14}  not scorable")
            continue
        obs, cv, n = r
        e, lo, hi = lam(obs)
        rt = e / (sum(-d * L for L, d in cv) / sum(L * L for L, _ in cv))
        rlo, rhi = ratio_ci(obs, cv)
        pooled_o += obs
        pooled_c += cv
        print(f"{con:<17}{obj:<14}{n:>7}{len(obs):>7}{e:>8.3f}"
              f"{rt:>10.2f} [{rlo:.2f}, {rhi:.2f}]")
    if pooled_o:
        e, lo, hi = lam(pooled_o)
        rt = e / (sum(-d * L for L, d in pooled_c) / sum(L * L for L, _ in pooled_c))
        rlo, rhi = ratio_ci(pooled_o, pooled_c)
        print(f"\n{'POOLED (new)':<31}{'':>7}{len(pooled_o):>7}{e:>8.3f}"
              f"{rt:>10.2f} [{rlo:.2f}, {rhi:.2f}]")
    print("\nnull under any Gaussian contest: ratio = 1.00 exactly")
    print("machines: lambda 0.690, ratio 5.77 [4.32, 7.65]")


if __name__ == "__main__":
    main()
