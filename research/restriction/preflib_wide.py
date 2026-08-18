"""Human odds invariance at scale, across construct types.

The first pass took ten complete-ranking files and pooled them. PrefLib holds
2,020 strict-complete and 312 tied-complete files, and they are not all the same
kind of thing: real election ballots, food and product preference, perceptual
discrimination, and sporting finish orders all appear. Pooling them answers no
clear question, so this harvests broadly and reports the discount

    delta_ij = -lambda * log(p_i/p_j)

separately by construct, alongside the Case V contest prediction computed from
each file's own unrestricted distribution. lambda = 0 is Luce's axiom, lambda = 1
abandons the prior ranking.

Tied-complete files are included: a voter whose top available group contains
several alternatives contributes equally to each, which is the standard reading
of an expressed tie.

Sporting finish orders are reported but must not be read as a test. A finish
order IS a race, so a race model fitting it is tautological rather than
informative; the row is a ceiling, showing what this statistic looks like when
the generating process is literally the model, and nothing about choice can be
inferred from it.

Usage:  python preflib_wide.py [max_files_per_collection]
"""
import json
import math
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

HERE = Path(__file__).parent
CACHE = HERE / "preflib"
API = "https://api.github.com/repos/PrefLib/PrefLib-Data/contents/datasets"
RAW = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets"

MIN_VOTERS, MIN_ALTS, MAX_ALTS = 100, 3, 12

CONSTRUCT = {
    "preference": {"sushi", "netflix", "breakfast", "shirt", "poster",
                   "boardgames", "movehub", "countries"},
    "election": {"apa", "vermont", "irish", "debian", "ers", "glasgow", "sf",
                 "oakland", "pierce", "sl", "takomapark", "minneapolis",
                 "burlington", "aspen", "berkley", "uklabor", "frenchapproval",
                 "frenchrate", "education", "agh", "project", "university"},
    "perception": {"dots", "puzzle"},
    "sport": {"boxing", "cycling", "tabletennis", "tennis", "mylaps", "skate"},
}


def construct_of(coll):
    tag = coll.split("-", 1)[-1].strip().lower()
    for k, v in CONSTRUCT.items():
        if tag in v:
            return k
    return None


def gj(u):
    """Listing via the gh CLI, which is authenticated: the unauthenticated
    GitHub API allows sixty requests an hour and this needs more."""
    import subprocess
    path = u.replace("https://api.github.com/", "")
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True,
                         timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:200])
    return json.loads(out.stdout)


def fetch(path):
    local = CACHE / path.replace("/", "__")
    if local.exists():
        return local.read_text()
    with urllib.request.urlopen(f"{RAW}/{urllib.parse.quote(path)}", timeout=90) as r:
        txt = r.read().decode("utf-8", "replace")
    CACHE.mkdir(parents=True, exist_ok=True)
    local.write_text(txt)
    return txt


def parse(txt):
    """Returns (title, n_alts, [(count, [group, ...])]) with groups as tuples."""
    title, n_alts, rows = "", None, []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# TITLE:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("# NUMBER ALTERNATIVES:"):
                try:
                    n_alts = int(line.split(":", 1)[1])
                except ValueError:
                    pass
            continue
        head, _, rest = line.partition(":")
        if not rest:
            head, _, rest = line.partition(",")
        try:
            cnt = int(head.strip())
        except ValueError:
            continue
        groups, buf, inbrace = [], "", False
        for ch in rest:
            if ch == "{":
                inbrace = True
                buf = ""
            elif ch == "}":
                inbrace = False
                groups.append(tuple(int(x) for x in buf.split(",") if x.strip()))
                buf = ""
            elif ch == "," and not inbrace:
                if buf.strip():
                    groups.append((int(buf),))
                buf = ""
            else:
                buf += ch
        if buf.strip():
            groups.append((int(buf),))
        groups = [g for g in groups if g]
        if groups:
            rows.append((cnt, groups))
    return title, n_alts, rows


def shares(rows, available):
    """Population shares, splitting an expressed tie equally."""
    c, n = defaultdict(float), 0.0
    for cnt, groups in rows:
        for g in groups:
            hit = [x for x in g if x in available]
            if hit:
                for x in hit:
                    c[x] += cnt / len(hit)
                n += cnt
                break
    return ({a: c[a] / n for a in available}, n) if n else (None, 0)


def lam(pairs, seed=4, B=8000):
    num = sum(-d * L for L, d in pairs)
    den = sum(L * L for L, _ in pairs)
    if den <= 0:
        return None
    est = num / den
    random.seed(seed)
    n, bs = len(pairs), []
    for _ in range(B):
        smp = [pairs[random.randrange(n)] for _ in range(n)]
        de = sum(L * L for L, _ in smp)
        if de > 0:
            bs.append(sum(-d * L for L, d in smp) / de)
    bs.sort()
    return est, bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]


def analyse(rows, n_alts):
    alts = set(range(1, n_alts + 1))
    full, n = shares(rows, alts)
    if not full or n < MIN_VOTERS:
        return None
    live = sorted([a for a in alts if full[a] > 0])
    if len(live) < MIN_ALTS:
        return None
    p = [full[a] for a in live]
    z = sum(p)
    p = [x / z for x in p]
    try:
        a_loc, err = calibrate_np(p)
    except Exception:
        return None
    if err > 0.05:
        return None
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pair, m = shares(rows, {i, j})
            if not pair or pair.get(i, 0) <= 0 or pair.get(j, 0) <= 0:
                continue
            L = math.log(full[i] / full[j])
            if abs(L) < 1e-6:
                continue
            obs.append((L, math.log(pair[i] / pair[j]) - L))
            w = win_probs_np(a_loc[[x, y]])
            if w[0] > 0 and w[1] > 0:
                cv.append((L, math.log(w[0] / w[1]) - L))
    return (obs, cv, n) if len(obs) >= 3 else None


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    dirs = [x["name"] for x in gj(API) if x["type"] == "dir"]
    by = defaultdict(lambda: {"obs": [], "cv": [], "files": 0, "voters": 0})
    for d in dirs:
        con = construct_of(d)
        if not con:
            continue
        try:
            files = [f["name"] for f in gj(f"{API}/{urllib.parse.quote(d)}")]
        except Exception:
            continue
        cand = [f for f in files if f.endswith((".soc", ".toc"))]
        used = 0
        for fn in cand:
            if used >= cap:
                break
            try:
                title, K, rows = parse(fetch(f"{d}/{fn}"))
            except Exception:
                continue
            if not K or not (MIN_ALTS <= K <= MAX_ALTS):
                continue
            r = analyse(rows, K)
            if not r:
                continue
            obs, cv, n = r
            by[con]["obs"].extend(obs)
            by[con]["cv"].extend(cv)
            by[con]["files"] += 1
            by[con]["voters"] += n
            used += 1
        if used:
            print(f"  {d[:28]:<30} {con:<11} {used} files", flush=True)

    print(f"\n{'construct':<13}{'files':>6}{'voters':>10}{'pairs':>7}"
          f"{'observed lambda':>22}{'Case V lambda':>20}")
    out = {}
    for con, v in sorted(by.items()):
        lo_ = lam(v["obs"])
        lc = lam(v["cv"]) if v["cv"] else None
        if not lo_:
            continue
        out[con] = {"files": v["files"], "voters": v["voters"],
                    "pairs": len(v["obs"]), "observed": lo_, "casev": lc}
        cvs = f"{lc[0]:.3f} [{lc[1]:.3f}, {lc[2]:.3f}]" if lc else "-"
        print(f"{con:<13}{v['files']:>6}{v['voters']:>10}{len(v['obs']):>7}"
              f"{lo_[0]:>10.3f} [{lo_[1]:.3f}, {lo_[2]:.3f}]{cvs:>20}")
    (HERE / "preflib_wide_results.json").write_text(json.dumps(
        {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
             for kk, vv in v.items()} for k, v in out.items()}, indent=1))
    print("\nmachines, same statistic: observed 0.690 [0.585, 0.792], "
          "Case V 0.120 [0.095, 0.147]")
    print("Luce requires lambda = 0; lambda = 1 discards the prior ranking")


if __name__ == "__main__":
    main()
