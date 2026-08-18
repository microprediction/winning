"""Twelve APA electorates: is the outlier a property of the format or an accident?

The 1980 APA presidential ballot gave a discount ratio of 3.49, one of three
values in the human panel that sit well above the null of one. With a single
electorate there is no way to tell a real feature from an accident, but PrefLib
carries twelve APA elections between 1998 and 2009, each five candidates ranked
without ties by a separate electorate of tens of thousands.

Twelve independent replications settle it. If preferential ballots really produce
a large discount, all twelve should; if 1980 was an accident, the twelve should
scatter around one.

Ballots are truncated at the voter's discretion, so only complete rankings are
used, which is a self-selected subpopulation and is stated as a limitation rather
than corrected.

Usage:  python apa_multi.py
"""
import math
import random
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

CACHE = Path(__file__).parent / "preflib"
RAW = ("https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/"
       "00028%20-%20apa")


def fetch(name):
    local = CACHE / f"apa__{name}"
    if local.exists():
        txt = local.read_text()
        # an earlier run may have cached a 404 body; treat it as absent
        if "# DATA TYPE" in txt:
            return txt
        local.unlink()
    out = subprocess.run(["curl", "-sL", "--max-time", "60", f"{RAW}/{name}"],
                         capture_output=True, text=True)
    txt = out.stdout
    # a missing path returns an HTML or plain "404: Not Found" body, not a file
    if out.returncode != 0 or not txt.strip() or "# DATA TYPE" not in txt:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    local.write_text(out.stdout)
    return out.stdout


def parse_complete(txt, K=5):
    """Complete strict ballots only: exactly K distinct candidates, no ties."""
    rows, title = [], ""
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("#"):
            if line.startswith("# TITLE:"):
                title = line.split(":", 1)[1].strip()
            continue
        head, _, rest = line.partition(":")
        try:
            cnt = int(head)
        except ValueError:
            continue
        if "{" in rest:
            continue                        # any tie group disqualifies
        order = [int(x) for x in rest.split(",") if x.strip()]
        if len(order) == K and len(set(order)) == K:
            rows.append((cnt, order))
    return title, rows


def shares(rows, avail):
    c = {a: 0 for a in avail}
    n = 0
    for cnt, order in rows:
        for it in order:
            if it in avail:
                c[it] += cnt
                n += cnt
                break
    return ({a: c[a] / n for a in avail}, n) if n else (None, 0)


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


def analyse(rows, K=5):
    alts = set(range(1, K + 1))
    full, n = shares(rows, alts)
    if not full or n < 500:
        return None
    live = sorted([a for a in alts if full[a] > 0])
    if len(live) < 3:
        return None
    p = [full[a] for a in live]
    z = sum(p)
    a_loc, err = calibrate_np([x / z for x in p])
    if err > 0.05:
        return None
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pr, m = shares(rows, {i, j})
            if not pr or pr[i] <= 0 or pr[j] <= 0:
                continue
            L = math.log(full[i] / full[j])
            if abs(L) < 1e-6:
                continue
            w = win_probs_np(a_loc[[x, y]])
            if w[0] <= 0 or w[1] <= 0:
                continue
            obs.append((L, math.log(pr[i] / pr[j]) - L))
            cv.append((L, math.log(w[0] / w[1]) - L))
    return (obs, cv, n) if len(obs) >= 3 else None


def main():
    print(f"{'election':<22}{'ballots':>9}{'pairs':>7}{'lambda':>8}"
          f"{'ratio':>8}{'95% CI':>18}")
    allo, allc, ratios = [], [], []
    for k in range(1, 14):
        for ext in ("soi", "toc", "soc"):
            txt = fetch(f"00028-{k:08d}.{ext}")
            if txt:
                break
        if not txt:
            continue
        title, rows = parse_complete(txt)
        if not rows:
            continue
        r = analyse(rows)
        if not r:
            print(f"{title[:21]:<22}  not scorable")
            continue
        obs, cv, n = r
        rt = lam(obs) / lam(cv)
        lo, hi = ratio_ci(obs, cv)
        allo += obs
        allc += cv
        ratios.append(rt)
        print(f"{title[:21]:<22}{n:>9}{len(obs):>7}{lam(obs):>8.3f}"
              f"{rt:>8.2f}   [{lo:.2f}, {hi:.2f}]")
    if allo:
        rt = lam(allo) / lam(allc)
        lo, hi = ratio_ci(allo, allc)
        print(f"\n{'POOLED':<22}{'':>9}{len(allo):>7}{lam(allo):>8.3f}"
              f"{rt:>8.2f}   [{lo:.2f}, {hi:.2f}]")
        print(f"  individual ratios: "
              f"{', '.join(f'{x:.2f}' for x in sorted(ratios))}")
    print("\n  null 1.00; the 1980 APA ballot gave 3.49; machines 5.77")


if __name__ == "__main__":
    main()
