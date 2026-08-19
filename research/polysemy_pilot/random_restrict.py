"""Random-elicitation restriction test with exact logprobs (OpenAI).

Design per (category, phrasing, model):
  U:  "Name a random metal."                       -> p over inventory
  A:  "Name a random metal that is not gold."      -> exclusion variant
  B:  "I already picked one random metal: gold.
       Name another random metal, different from gold."  -> two-slot variant

where 'gold' is that cell's own modal item under U. Zero-parameter
predictions of the restricted distribution over inventory-minus-favorite:
Luce renormalization vs Thurstone contestant removal calibrated on U.
"Random" elicitation restores entropy that preference-tuned models lack
under "favorite/best" prompts, mirroring Cotton (2024)'s explicit
single-item elimination design.
"""
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, SYSTEM, key, match_items
from exact_analyze import INVENTORY, calibrate_np, win_probs_np, rmse, entropy_norm

from openai import OpenAI

CLIENT = OpenAI(api_key=key())
PHRASES = ["Name", "Pick"]
NOUNS = list(INVENTORY.keys())  # 17 categories


def top20(prompt: str, model: str) -> dict[str, float]:
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0,
        messages=[{"role": "system", "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def inv_dist(raw_dist, noun):
    inv = set(INVENTORY[noun])
    return {k: v for k, v in match_items(raw_dist).items() if k in inv}


def main():
    # --- stage 1: unqualified
    jobs = [(n, ph, m) for n in NOUNS for ph in PHRASES for m in MODELS]
    unq = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(top20, f"{ph} a random {n}.", m): (n, ph, m)
                for n, ph, m in jobs}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                unq[k] = fut.result()
            except Exception as e:
                print(f"ERROR U {k}: {e}", file=sys.stderr, flush=True)
    print(f"unqualified done: {len(unq)}", flush=True)

    # --- stage 2: restricted (favorite determined per cell)
    plans = []
    for (n, ph, m), raw in unq.items():
        d = inv_dist(raw, n)
        if len(d) < 3:
            continue
        fav = max(d, key=d.get)
        pa = f"{ph} a random {n} that is not {fav}."
        pb = (f"I already picked one random {n}: {fav}. "
              f"{ph} another random {n}, different from {fav}.")
        plans.append((n, ph, m, fav, "exclusion", pa))
        plans.append((n, ph, m, fav, "two-slot", pb))

    restr = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(top20, p, m): (n, ph, m, fav, var)
                for n, ph, m, fav, var, p in plans}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                restr[k] = fut.result()
            except Exception as e:
                print(f"ERROR R {k}: {e}", file=sys.stderr, flush=True)
    print(f"restricted done: {len(restr)}", flush=True)

    json_safe = {f"{n}||{ph}||{m}": raw for (n, ph, m), raw in unq.items()}
    json_safe |= {f"{n}||{ph}||{m}||{fav}||{var}": raw
                  for (n, ph, m, fav, var), raw in restr.items()}
    (HERE / "random_raw.json").write_text(json.dumps(json_safe, indent=1))

    # --- stage 3: analysis
    results = []
    for (n, ph, m, fav, var), raw_q in restr.items():
        d_u = inv_dist(unq[(n, ph, m)], n)
        d_q = inv_dist(raw_q, n)
        keep = [s for s in d_u if s != fav and s in d_q]
        if len(keep) < 2:
            continue

        az = sum(d_q[s] for s in keep)
        actual = [d_q[s] / az for s in keep]

        items = sorted(d_u, key=d_u.get, reverse=True)
        uz = sum(d_u.values())
        p_full = [d_u[s] / uz for s in items]

        lz = sum(d_u[s] for s in keep)
        luce_pred = [d_u[s] / lz for s in keep]

        a, cal_err = calibrate_np(p_full)
        if cal_err > 0.05:
            continue
        idx = [items.index(s) for s in keep]
        tp = win_probs_np(a[idx])
        thur_pred = (tp / tp.sum()).tolist()

        results.append({
            "cell": n, "phrasing": ph, "model": m, "variant": var,
            "favorite": fav, "keep": keep, "actual": actual,
            "luce": luce_pred, "thurstone": thur_pred,
            "rmse_luce": rmse(luce_pred, actual),
            "rmse_thurstone": rmse(thur_pred, actual),
            "H_unq": entropy_norm(p_full), "fav_p": max(p_full),
        })

    (HERE / "random_results.json").write_text(json.dumps(results, indent=1))
    print(f"\n{len(results)} usable cells")
    mean_H = sum(r["H_unq"] for r in results) / len(results)
    print(f"mean unqualified normalized entropy: {mean_H:.3f} "
          f"(vs near-degenerate 'favorite/best' prompts)")

    for grp_name, grp_key in [("by model", "model"), ("by variant", "variant")]:
        print(f"--- {grp_name}")
        for g in sorted({r[grp_key] for r in results}):
            rs = [r for r in results if r[grp_key] == g]
            tw = sum(r["rmse_thurstone"] < r["rmse_luce"] for r in rs)
            pl = rmse([x for r in rs for x in r["luce"]],
                      [x for r in rs for x in r["actual"]])
            pt = rmse([x for r in rs for x in r["thurstone"]],
                      [x for r in rs for x in r["actual"]])
            print(f"{g:<12} cells={len(rs):>3}  Thurstone wins {tw}/{len(rs)}  "
                  f"pooled RMSE: Luce={pl:.4f} Thurstone={pt:.4f}")

    diffs = [r["rmse_luce"] - r["rmse_thurstone"] for r in results]
    nn = len(diffs)
    random.seed(1234)
    B = 10000
    wins = sum(sum(diffs[random.randrange(nn)] for _ in range(nn)) > 0
               for _ in range(B))
    print(f"\nmean RMSE diff (Luce - Thurstone): {sum(diffs)/nn:+.4f}")
    print(f"bootstrap P(Thurstone better) = {wins/B:.4f}  (n={nn} cells)")


if __name__ == "__main__":
    main()
