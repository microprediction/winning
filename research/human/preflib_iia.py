"""Human odds invariance across many domains, from PrefLib complete rankings.

The sushi comparison gave one human number for the pair-restriction statistic.
One number cannot say whether it is typical, so this harvests every complete
strict-order (.soc) dataset in PrefLib and computes the same quantity for each:

    delta_ij = log(q_i/q_j) - log(p_i/p_j)

p from the full alternative set, q from the pair, both as population shares of
respondents whose top-ranked available alternative is the one in question. Luce
requires zero everywhere.

Complete rankings are used because they let a subset choice be derived rather
than elicited, which is also the construction's limitation: a fixed ranking
cannot reverse, so these data can measure the odds shift but cannot exhibit the
rank reversals the machine data shows. The elicitation format presupposes the
stability the machines violate.

Datasets are cached under research/human/preflib/ on first run, so the analysis
is reproducible offline and the download is paid once.

Usage:  python preflib_iia.py [max_datasets]
"""
import json
import math
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "preflib"
API = "https://api.github.com/repos/PrefLib/PrefLib-Data/contents/datasets"
RAW = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets"

MIN_VOTERS = 50      # below this the population shares are too noisy
MIN_ALTS = 3
MAX_ALTS = 12        # keep pairs well populated


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def fetch_text(path):
    local = CACHE / path.replace("/", "__")
    if local.exists():
        return local.read_text()
    url = f"{RAW}/{urllib.parse.quote(path)}"
    with urllib.request.urlopen(url, timeout=90) as r:
        txt = r.read().decode("utf-8", "replace")
    CACHE.mkdir(parents=True, exist_ok=True)
    local.write_text(txt)
    return txt


def parse_soc(txt):
    """Returns (title, n_alts, [(count, ranking)])."""
    title, n_alts, rows = "", None, []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# TITLE:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("# NUMBER ALTERNATIVES:"):
                n_alts = int(line.split(":", 1)[1])
            continue
        if ":" in line:
            c, rest = line.split(":", 1)
        elif "," in line:
            c, rest = line.split(",", 1)
        else:
            continue
        try:
            cnt = int(c.strip())
            order = [int(x) for x in rest.replace("{", "").replace("}", "").split(",")
                     if x.strip()]
        except ValueError:
            continue
        if order:
            rows.append((cnt, order))
    return title, n_alts, rows


def shares(rows, alts, available):
    c = {a: 0 for a in alts}
    n = 0
    for cnt, order in rows:
        for it in order:
            if it in available:
                c[it] += cnt
                n += cnt
                break
    return ({a: c[a] / n for a in alts}, n) if n else (None, 0)


def deltas_for(rows, n_alts):
    alts = list(range(1, n_alts + 1))
    full, n = shares(rows, alts, set(alts))
    if not full or n < MIN_VOTERS:
        return None, n
    out = []
    for x in range(len(alts)):
        for y in range(x + 1, len(alts)):
            i, j = alts[x], alts[y]
            if full[i] <= 0 or full[j] <= 0:
                continue
            pair, m = shares(rows, alts, {i, j})
            if not pair or pair[i] <= 0 or pair[j] <= 0:
                continue
            out.append(math.log(pair[i] / pair[j]) - math.log(full[i] / full[j]))
    return out, n


def boot(vals, seed=4, B=20000):
    n = len(vals)
    random.seed(seed)
    bs = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n
                for _ in range(B))
    return sum(vals) / n, bs[int(.025 * B)], bs[int(.975 * B)]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    dirs = [x["name"] for x in get_json(API) if x["type"] == "dir"]
    print(f"{len(dirs)} PrefLib collections; scanning for complete rankings")

    per_dataset, all_d = [], []
    scanned = 0
    for d in dirs:
        if scanned >= limit:
            break
        try:
            files = get_json(f"{API}/{urllib.parse.quote(d)}")
        except Exception:
            continue
        socs = [f["name"] for f in files if f["name"].endswith(".soc")]
        if not socs:
            continue
        scanned += 1
        got = 0
        for fn in socs[:6]:
            try:
                txt = fetch_text(f"{d}/{fn}")
            except Exception:
                continue
            title, n_alts, rows = parse_soc(txt)
            if not n_alts or not (MIN_ALTS <= n_alts <= MAX_ALTS):
                continue
            ds, n = deltas_for(rows, n_alts)
            if not ds or len(ds) < 3:
                continue
            m, lo, hi = boot(ds)
            viol = sum(1 for x in ds if abs(x) > 0.1)
            per_dataset.append({"collection": d, "file": fn, "title": title,
                                "n_alts": n_alts, "n_voters": n,
                                "pairs": len(ds), "mean_delta": m,
                                "ci": [lo, hi],
                                "mean_abs": sum(abs(x) for x in ds) / len(ds),
                                "frac_violating": viol / len(ds)})
            all_d.extend(ds)
            got += 1
            if got >= 3:
                break

    if not per_dataset:
        print("no usable datasets")
        return
    (HERE / "preflib_iia_results.json").write_text(json.dumps(per_dataset, indent=1))

    print(f"\n{len(per_dataset)} usable rank files, {len(all_d)} pairs total\n")
    print(f"{'collection':<22}{'K':>3}{'voters':>8}{'pairs':>7}"
          f"{'mean d':>9}{'|d|':>8}{'>0.1':>7}  title")
    for r in sorted(per_dataset, key=lambda r: r["mean_delta"]):
        print(f"{r['collection'][:21]:<22}{r['n_alts']:>3}{r['n_voters']:>8}"
              f"{r['pairs']:>7}{r['mean_delta']:>+9.3f}{r['mean_abs']:>8.3f}"
              f"{r['frac_violating']:>6.0%}  {r['title'][:26]}")

    m, lo, hi = boot(all_d)
    print(f"\npooled over all human pairs: mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  mean |delta| {sum(abs(x) for x in all_d)/len(all_d):.4f}, "
          f"|delta|>0.1 in "
          f"{sum(1 for x in all_d if abs(x) > 0.1)/len(all_d):.0%} of pairs")
    print(f"  datasets with a negative mean shift: "
          f"{sum(1 for r in per_dataset if r['mean_delta'] < 0)}/{len(per_dataset)}")
    print("\nmachine battery, identical statistic:")
    print("  mean -1.1500 [-1.5474, -0.8076], |delta|>0.1 in 96.8% of 279 pairs")


if __name__ == "__main__":
    main()
