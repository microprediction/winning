"""Run the original winningbook question volumes (Cotton 2024) on GPT models
with exact logprobs.

Uses the ORIGINAL prompt templates verbatim as cloze fills:
  U: Fill in the blank...: "My favourite organ is the ___ because it is vital."
  R: ... "My two favourite organs are the {fav} and the ___ because they are vital."
fav = argmax of U. Luce renormalization vs Thurstone contestant removal on the
open-vocabulary item set, KL-scored.
"""
import glob
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, key
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
N_ADJ = 0  # 0 = every adjective in the volume; >0 = random subsample of that size
STOP = set("the and for its this that with are was very a an of to in on it is "
           "because they most two favourite favorite".split())


def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def items_of(raw):
    out = {}
    for tok, p in raw.items():
        w = tok.strip().lower()
        if len(w) >= 3 and w.isalpha() and w not in STOP:
            out[w] = out.get(w, 0.0) + p
    return out


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-9)) for ai, pi in zip(a, p) if ai > 0)


def load_categories():
    cats = {}
    for f in sorted(glob.glob(str(HERE / "qvols" / "*.yaml"))):
        cats.update(yaml.safe_load(open(f)).get("examples", {}))
    return cats


def run_cell(cat, spec, adj, model):
    t_orig, t_qual = spec["prompt_pair_template"][:2]
    s_orig = t_orig.replace("[MASK]", "___").replace("SOMETHING", adj)
    u_raw = top20(f'Fill in the blank with a single word: "{s_orig}" '
                  "Give only the missing word.", model)
    d_u = items_of(u_raw)
    if len(d_u) < 3:
        return None
    fav = max(d_u, key=d_u.get)
    s_qual = (t_qual.replace("[ANSWER]", fav).replace("[MASK]", "___")
              .replace("SOMETHING", adj))
    q_raw = top20(f'Fill in the blank with a single word: "{s_qual}" '
                  "Give only the missing word.", model)
    d_q = items_of(q_raw)

    # calibration field: top-10 unqualified items
    items = sorted(d_u, key=d_u.get, reverse=True)[:10]
    uz = sum(d_u[s] for s in items)
    p_full = [d_u[s] / uz for s in items]

    keep = [s for s in items if s != fav and s in d_q]
    if len(keep) < 2:
        return None
    az = sum(d_q[s] for s in keep)
    actual = [d_q[s] / az for s in keep]

    lz = sum(d_u[s] for s in keep)
    luce = [d_u[s] / lz for s in keep]

    a, err = calibrate_np(p_full)
    if err > 0.05:
        return None
    idx = [items.index(s) for s in keep]
    w = win_probs_np(a[idx])
    thur = (w / w.sum()).tolist()

    return {"category": cat, "adjective": adj, "model": model, "fav": fav,
            "keep": keep, "actual": actual, "luce": luce, "thurstone": thur,
            "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
            "H_unq": entropy_norm(p_full)}


def main():
    cats = load_categories()
    rng = random.Random(2024)
    jobs = []
    for cat, spec in cats.items():
        adjs = spec.get("adjectives", [])
        if not adjs or "prompt_pair_template" not in spec:
            continue
        use = adjs if N_ADJ == 0 else rng.sample(adjs, min(N_ADJ, len(adjs)))
        for adj in use:
            for m in MODELS:
                jobs.append((cat, spec, adj, m))
    print(f"{len(jobs)} cells", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run_cell, *j) for j in jobs]
        for k, fut in enumerate(as_completed(futs)):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 100 == 0:
                print(f"{k+1}/{len(jobs)}", flush=True)

    (HERE / "vol_battery_results.json").write_text(json.dumps(results, indent=1))
    n = len(results)
    print(f"\n{n} usable cells (of {len(jobs)})")

    def rep(name, ss):
        if not ss:
            return
        nn = len(ss)
        diffs = [r["kl_luce"] - r["kl_thur"] for r in ss]
        mean = sum(diffs) / nn
        random.seed(4)
        B = 20000
        boots = sorted(sum(diffs[random.randrange(nn)] for _ in range(nn)) / nn
                       for _ in range(B))
        tw = sum(d > 0 for d in diffs)
        print(f"{name:<30} n={nn:>4} Thurstone wins {tw:>4}/{nn}  "
              f"mean dKL={mean:+.4f} [{boots[int(.025*B)]:+.4f},{boots[int(.975*B)]:+.4f}]  "
              f"P(Luce better)={sum(b <= 0 for b in boots)/B:.4f}")

    rep("ALL (original 2024 stimuli)", results)
    for m in MODELS:
        rep(f"  {m}", [r for r in results if r["model"] == m])
    rep("  non-degenerate (H>0.2)", [r for r in results if r["H_unq"] > 0.2])
    rep("  degenerate (H<=0.2)", [r for r in results if r["H_unq"] <= 0.2])


if __name__ == "__main__":
    main()
