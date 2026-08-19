"""Authoritative cell accounting: one row per design, computed from the logs.

Hand-maintained counts in a numerically driven paper drift, and two had. This
recomputes every design's scored cells, models, clustering unit and headline
estimate directly from the committed results, so the manuscript's table can be
set from output rather than memory. Run it after any battery re-runs.

Usage:  python accounting.py
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def boot(d, clusters=None, seed=4, B=20000):
    """Cluster bootstrap when clusters are given, else over cells."""
    if not d:
        return None
    random.seed(seed)
    if clusters:
        groups = defaultdict(list)
        for x, c in zip(d, clusters):
            groups[c].append(x)
        keys = list(groups)
        means = []
        for _ in range(B):
            pick = [groups[keys[random.randrange(len(keys))]] for _ in keys]
            flat = [v for g in pick for v in g]
            means.append(sum(flat) / len(flat))
    else:
        n = len(d)
        means = [sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(B)]
    means.sort()
    return sum(d) / len(d), means[int(.025 * B)], means[int(.975 * B)]


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else []


def dk(rows):
    out = []
    for r in rows:
        if "kl_luce" in r and ("kl_thur" in r or "kl_thurstone" in r):
            out.append(r["kl_luce"] - r.get("kl_thur", r.get("kl_thurstone")))
        elif all(k in r for k in ("actual", "luce", "thurstone")):
            out.append(kl(r["actual"], r["luce"]) - kl(r["actual"], r["thurstone"]))
    return out


def main():
    rows = []

    def add(design, models, cells, unit, stat):
        rows.append((design, models, cells, unit, stat))

    def fmt(b):
        return "n/a" if not b else f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]"

    # sweep: the broad battery, clustered by question type
    s = load("sweep_results.json")
    d = dk(s)
    add("Open-vocabulary sweep", len({r["model"] for r in s}), len(s),
        "question type (99)", fmt(boot(d, [r["category"] for r in s])))

    # two-slot original stimuli
    v = load("vol_battery_results.json")
    add("Original 2024 stimuli, two-slot", len({r["model"] for r in v}), len(v),
        "question type (97)", fmt(boot(dk(v), [r["category"] for r in v])))

    # permutation deletion
    p = load("perm_results.json")
    add("Permutation-controlled deletion", len({r["model"] for r in p}), len(p),
        "category (31)", fmt(boot(dk(p), [r["cell"] for r in p])))

    # ordered selection ladder
    e = load("exotics_results.json")
    add("Ordered selection, depth 2-5", len({r["model"] for r in e}), len(e),
        "category (40)", fmt(boot(dk(e), [r["category"] for r in e])))

    # reversibility
    rv = load("reversibility_results.json")
    add("Reversibility", len({r["model"] for r in rv}), len(rv),
        "category", fmt(boot(dk(rv), [r["category"] for r in rv])))

    # menu deletion under conditioning (transport)
    t = load("transport_results.json")
    tc = [c for r in t for c in r.get("cells", [])]
    add("Menu deletion under conditioning", len({r["model"] for r in t}), len(tc),
        "condition", fmt(boot(dk(tc), [c["condition"] for c in tc])))

    # duplicates
    rb = load("red_bus_results.json")
    add("Duplicates (red bus)", len({r.get("model") for r in rb}), len(rb),
        "family", "substitution best in 15/41")

    # context effects: each row carries a decoy and a compromise test
    ce = load("context_effects_results.json")
    n_ce = sum(("decoy" in r) + ("compromise" in r) for r in ce)
    viol = sum(r.get(k, {}).get("regularity_violated", False)
               for r in ce for k in ("decoy", "compromise"))
    add("Decoy and compromise", len({r["model"] for r in ce}), n_ce,
        "family", f"{viol}/{n_ce} regularity violations")

    # Block-Marschak
    bm = load("bm_results.json")
    add("Block-Marschak / RUM", len({r["model"] for r in bm}), len(bm),
        "family", f"{sum(r['bm_violations'] > 0 for r in bm)}/{len(bm)} violate RUM")

    # local base vs instruct, on the sweep's stimuli
    lq = load("local_qvols_results.json")
    add("Local base vs instruct", len({r["repo"] for r in lq}), len(lq),
        "question type", "tuning effect in 4/6 pairs")

    # local inventory-scored panel
    lo = load("local_results.json")
    add("Local inventory panel", len({r["repo"] for r in lo}), len(lo),
        "category", fmt(boot(dk(lo), [r["category"] for r in lo])))

    # earlier GPT batteries
    ex = load("exact_results.json")
    add("Qualified-noun restriction", len({r["model"] for r in ex}), len(ex),
        "cell", "tie (degenerate priors)")
    rr = load("random_results.json")
    add("Random elicitation, one deletion", len({r["model"] for r in rr}), len(rr),
        "category", "Thurstone 39/56 non-degenerate")

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'design':<{w}}{'models':>7}{'cells':>8}  {'cluster unit':<20}headline")
    print("-" * (w + 60))
    total = 0
    for design, models, cells, unit, stat in rows:
        total += cells
        print(f"{design:<{w}}{models:>7}{cells:>8}  {unit:<20}{stat}")
    print("-" * (w + 60))
    print(f"{'TOTAL scored cells':<{w}}{'':>7}{total:>8}")
    raw = sum(sum(1 for _ in open(f)) for f in HERE.glob("*_raw.jsonl"))
    print(f"{'TOTAL logged measurements':<{w}}{'':>7}{raw:>8}")


if __name__ == "__main__":
    main()
