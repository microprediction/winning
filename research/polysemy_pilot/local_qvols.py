"""Local models on the sweep's own stimuli, for an apples-to-apples comparison.

Two explanations compete for the gap between the API panel and the local panel.
Either capability drives the Thurstonian deficit, in which case small
open-weight models genuinely sit closer to Luce, or the stimuli do, since the
sweep uses the 2024 appendix templates with an adjective and a trailing clause
("My two favourite human organs are the heart and the ___ because they are
vital") while the local batteries used bare noun prompts.

Switching the local field to open-vocabulary did not close the gap, which rules
out the item-field explanation but not the stimulus explanation. This script
removes the remaining difference by running local models on exactly the prompts
sweep.py sends, scored the same way, so the only thing left varying is the
model.

Usage:  python local_qvols.py [n_adjectives] [model substring ...]
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from datastore import append_jsonl, load_jsonl, write_json_atomic
from local_survey import Scorer, MODELS, STOP, TOP_K, MAX_ITEMS
from sweep import load_categories, prompts, kl

import numpy as np

RAW_LOG = HERE / "local_qvols_raw.jsonl"


def field_from(pr, tok, max_items):
    agg = {}
    for idx in np.argsort(pr)[-500:][::-1]:
        w = tok.decode([int(idx)]).strip().lower()
        if len(w) >= 3 and w.isalpha() and w not in STOP:
            agg[w] = agg.get(w, 0.0) + float(pr[idx])
    items = sorted(agg, key=agg.get, reverse=True)[:max_items]
    return items, {i: agg[i] for i in items}


def restricted_from(pr, tok, items):
    agg = {i: 0.0 for i in items}
    for idx in np.argsort(pr)[-500:][::-1]:
        w = tok.decode([int(idx)]).strip().lower()
        if w in agg:
            agg[w] += float(pr[idx])
    return agg


def run(repo, family, size, tuning, cats, n_adj, cache):
    units = [(c, s, adj) for c, s in cats.items()
             for adj in s["adjectives"][:n_adj]]
    todo = [u for u in units if f"{repo}||{u[0]}||{u[2]}" not in cache]
    if not todo:
        print(f"  {repo}: fully cached", flush=True)
        return
    print(f"  loading {repo} ({len(todo)} units) ...", flush=True)
    try:
        sc = Scorer(repo)
    except Exception as e:
        print(f"  SKIP {repo}: {str(e)[:110]}", flush=True)
        return
    for k, (cat, spec, adj) in enumerate(todo):
        unq_prompt, _ = prompts(spec, adj, "x")
        pr = sc.next_token_probs(unq_prompt)
        items, d_u = field_from(pr, sc.tok, MAX_ITEMS)
        rec = {"key": f"{repo}||{cat}||{adj}", "repo": repo, "family": family,
               "size": size, "tuning": tuning, "category": cat,
               "adjective": adj, "items": items,
               "unqualified": [d_u[i] for i in items], "restricted": {}}
        if len(items) >= 3:
            for deleted in items[:TOP_K]:
                _, designs = prompts(spec, adj, deleted)
                prq = sc.next_token_probs(designs["two-slot"])
                agg = restricted_from(prq, sc.tok, items)
                rec["restricted"][deleted] = [agg[i] for i in items]
        append_jsonl(RAW_LOG, rec)
        cache[rec["key"]] = rec
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{len(todo)}", flush=True)


def score(rec):
    items = rec["items"]
    p = rec["unqualified"]
    if len(items) < 3 or sum(p) <= 0:
        return []
    z = sum(p)
    p_full = [x / z for x in p]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return []
    H = entropy_norm(p_full)
    rows = []
    for deleted, raw in rec["restricted"].items():
        di = items.index(deleted)
        keep = [i for i in range(len(items)) if i != di]
        az, lz = sum(raw[i] for i in keep), sum(p_full[i] for i in keep)
        if az <= 0 or lz <= 0:
            continue
        actual = [raw[i] / az for i in keep]
        luce = [p_full[i] / lz for i in keep]
        w = win_probs_np(a[keep])
        thur = (w / w.sum()).tolist()
        rows.append({"repo": rec["repo"], "family": rec["family"],
                     "size": rec["size"], "tuning": rec["tuning"],
                     "category": rec["category"], "adjective": rec["adjective"],
                     "deleted": deleted, "deleted_p": p_full[di],
                     "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
                     "H_unq": H})
    return rows


def main():
    import random
    n_adj = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    want = [a for a in sys.argv[2:]]
    cats = load_categories()
    cache = load_jsonl(RAW_LOG, key="key")
    todo = [m for m in MODELS if not want or any(w in m[0] for w in want)]
    print(f"{len(todo)} models, {len(cats)} question types, "
          f"{n_adj} adjectives each, {len(cache)} cached")
    for repo, family, size, tuning in todo:
        run(repo, family, size, tuning, cats, n_adj, cache)

    rows = []
    for rec in cache.values():
        rows.extend(score(rec))
    write_json_atomic(HERE / "local_qvols_results.json", rows)
    print(f"\n{len(rows)} scored cells on the sweep's own stimuli\n")

    def stat(ss):
        d = [r["kl_luce"] - r["kl_thur"] for r in ss]
        n = len(d)
        random.seed(4)
        B = 8000
        bs = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        return sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B)], n

    byrepo = {}
    for r in rows:
        byrepo.setdefault(r["repo"], []).append(r)
    print(f"{'model':<48}{'size':>6}{'n':>7}{'dKL':>9}  95% CI")
    for repo, ss in sorted(byrepo.items(), key=lambda kv: kv[1][0]["size"]):
        m, lo, hi, n = stat(ss)
        print(f"{repo[:47]:<48}{ss[0]['size']:>5.1f}B{n:>7}{m:>+9.3f}  "
              f"[{lo:+.3f},{hi:+.3f}]")
    print("\nFor reference, the same stimuli and design on API models:")
    print("  gpt-4.1-nano +0.111, gpt-4o-mini +0.140, gpt-4.1-mini +0.198"
          " (sweep, both designs pooled)")


if __name__ == "__main__":
    main()
