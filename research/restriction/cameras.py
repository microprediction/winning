"""Complete choice structures from humans: the matched comparison for the machines.

Costa-Gomes, Cueva, Gerasimou and Tejiscak (2022, Quantitative Economics) had 373
subjects choose from EVERY subset of five real digital cameras: 31 menus for the
first experiment, 26 for the second, each subject traversing the whole design once.
That is the human counterpart of the complete choice structures measured on
language models, and it supports three comparisons the ranking data could not.

  1. Order stability. Ranking data cannot show a reversal, because a subset choice
     derived from a fixed ranking preserves order by construction. Here the choices
     are separately elicited, so a reversal is observable: choose i from a menu and
     j from a submenu that still contains i.

  2. Random utility representability. Pooling subjects gives population choice
     probabilities for all 31 menus, which is exactly what the Block-Marschak
     conditions test, and exactly what the machine menu battery measured.

  3. The odds discount lambda, from the full menu down to each pair, at the
     population level.

Deferral is possible for some subjects, so the forced-choice subgroup is reported
separately: those subjects never defer, and no deferral-coding decision enters.

Data: the supplement zip of QE1806, public and without registration.

Usage:  python cameras.py path/to/Exp1_2_PREST.csv
"""
import itertools
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

ITEMS = ["Ca", "Hi", "Pa", "Sa", "So"]


def load(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            menu = tuple(sorted(x.strip() for x in r["menu"].split(",") if x.strip()))
            ch = (r["choice"] or "").strip()
            rows.append({"subject": r["subject"], "fc": r["fc"],
                         "menu": menu, "choice": ch or None})
    return rows


def reversals(rows):
    """WARP violations: chose i from A, then j from B subset A with i still in B."""
    by = defaultdict(dict)
    for r in rows:
        if r["choice"]:
            by[r["subject"]][r["menu"]] = r["choice"]
    tested = viol = 0
    subs_viol = set()
    for s, m in by.items():
        for A, cA in m.items():
            for B, cB in m.items():
                if len(B) >= len(A) or not set(B) < set(A):
                    continue
                if cA not in B:
                    continue           # the chosen item is gone: not a test
                tested += 1
                if cB != cA:
                    viol += 1
                    subs_viol.add(s)
    return tested, viol, len(subs_viol), len(by)


def population_structure(rows):
    """Choice probabilities per menu, pooled over subjects."""
    cnt = defaultdict(lambda: defaultdict(float))
    tot = defaultdict(float)
    for r in rows:
        if r["choice"]:
            cnt[r["menu"]][r["choice"]] += 1
            tot[r["menu"]] += 1
    return {m: {i: cnt[m][i] / tot[m] for i in m} for m in cnt if tot[m] >= 30}


def block_marschak(q, items):
    """K(x,S) = sum over supersets T of S of (-1)^{|T|-|S|} q(x,T); singletons = 1."""
    U = set(items)
    def qq(x, T):
        T = tuple(sorted(T))
        if len(T) == 1:
            return 1.0
        return q.get(T, {}).get(x, None)
    out = {}
    for x in items:
        for r in range(1, len(items) + 1):
            for S in itertools.combinations(items, r):
                if x not in S:
                    continue
                rest = list(U - set(S))
                tot, ok = 0.0, True
                for k in range(len(rest) + 1):
                    for extra in itertools.combinations(rest, k):
                        v = qq(x, set(S) | set(extra))
                        if v is None:
                            ok = False
                            break
                        tot += (-1) ** k * v
                    if not ok:
                        break
                if ok:
                    out[(x, S)] = tot
    return out


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


def discount(q):
    full = tuple(sorted(ITEMS))
    if full not in q:
        return None
    p = q[full]
    live = [i for i in ITEMS if p.get(i, 0) > 0]
    if len(live) < 3:
        return None
    vec = [p[i] for i in live]
    z = sum(vec)
    a_loc, err = calibrate_np([x / z for x in vec])
    if err > 0.05:
        return None
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pair = q.get(tuple(sorted((i, j))))
            if not pair or pair.get(i, 0) <= 0 or pair.get(j, 0) <= 0:
                continue
            L = math.log(p[i] / p[j])
            if abs(L) < 1e-6:
                continue
            w = win_probs_np(a_loc[[x, y]])
            if w[0] <= 0 or w[1] <= 0:
                continue
            obs.append((L, math.log(pair[i] / pair[j]) - L))
            cv.append((L, math.log(w[0] / w[1]) - L))
    return (obs, cv) if len(obs) >= 3 else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Exp1_2_PREST.csv"
    rows = load(path)
    for label, sel in (("all subjects", rows),
                       ("forced choice only", [r for r in rows if r["fc"] == "1"])):
        print(f"\n=== {label} ===")
        t, v, sv, ns = reversals(sel)
        print(f"  order stability: {v:,} reversals in {t:,} nested tests "
              f"({100*v/t:.1f}%), affecting {sv}/{ns} subjects")
        q = population_structure(sel)
        print(f"  population structure: {len(q)} menus with 30+ choices")
        K = block_marschak(q, ITEMS)
        if K:
            neg = [k for k, x in K.items() if x < -0.02]
            print(f"  Block-Marschak: {len(neg)} of {len(K)} sums negative "
                  f"beyond -0.02, worst {min(K.values()):+.3f}")
        d = discount(q)
        if d:
            obs, cv = d
            rt = lam(obs) / lam(cv)
            lo, hi = ratio_ci(obs, cv)
            print(f"  discount lambda {lam(obs):.3f}, ratio to Case V "
                  f"{rt:.2f} [{lo:.2f}, {hi:.2f}]  ({len(obs)} pairs)")
    print("\nmachine comparison, same three tests:")
    print("  order stability   32.8% of survivor pairs discordant, 86% of cells")
    print("  Block-Marschak    all 30 measured structures violate the conditions")
    print("  discount ratio    5.77 [4.32, 7.65]")


if __name__ == "__main__":
    main()
