"""Permutation-controlled deletion: for each category/model/phrasing, delete
EACH observed inventory item in turn (not just the favorite) and compare
Luce renormalization vs Thurstone contestant removal, exact logprobs."""
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, key, match_items
from exact_analyze import INVENTORY, calibrate_np, win_probs_np, rmse, entropy_norm
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
PHRASES = ["Name", "Pick"]
MAX_DELETIONS = 6  # delete each of the top-6 observed items in turn


def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "system", "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def inv_dist(raw, noun):
    inv = set(INVENTORY[noun])
    return {k: v for k, v in match_items(raw).items() if k in inv}


def main():
    unq = json.loads((HERE / "random_raw.json").read_text())  # reuse cached unqualified
    plans = []
    for n in INVENTORY:
        for ph in PHRASES:
            for m in MODELS:
                ku = f"{n}||{ph}||{m}"
                if ku not in unq:
                    continue
                d = inv_dist(unq[ku], n)
                if len(d) < 3:
                    continue
                items = sorted(d, key=d.get, reverse=True)[:MAX_DELETIONS]
                for out in items:
                    plans.append((n, ph, m, out,
                                  f"{ph} a random {n} that is not {out}."))
    # cache: skip deletions already fetched
    restr = {}
    cache_f = HERE / "perm_raw.json"
    if cache_f.exists():
        for k, v in json.loads(cache_f.read_text()).items():
            restr[tuple(k.split("||"))] = v
    plans = [p for p in plans if (p[0], p[1], p[2], p[3]) not in restr]
    print(f"{len(plans)} deletion calls ({len(restr)} cached)", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(top20, p, m): (n, ph, m, out)
                for n, ph, m, out, p in plans}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                restr[k] = fut.result()
            except Exception as e:
                print(f"ERROR {k}: {e}", file=sys.stderr, flush=True)
    (HERE / "perm_raw.json").write_text(json.dumps(
        {"||".join(k): v for k, v in restr.items()}, indent=1))

    results = []
    cal_cache = {}
    for (n, ph, m, out), raw_q in restr.items():
        d_u = inv_dist(unq[f"{n}||{ph}||{m}"], n)
        d_q = inv_dist(raw_q, n)
        keep = [s for s in d_u if s != out and s in d_q]
        if len(keep) < 2:
            continue
        az = sum(d_q[s] for s in keep)
        actual = [d_q[s] / az for s in keep]

        ck = f"{n}||{ph}||{m}"
        if ck not in cal_cache:
            items = sorted(d_u, key=d_u.get, reverse=True)
            uz = sum(d_u.values())
            a, err = calibrate_np([d_u[s] / uz for s in items])
            cal_cache[ck] = (items, a, err)
        items, a, err = cal_cache[ck]
        if err > 0.05:
            continue

        lz = sum(d_u[s] for s in keep)
        luce_pred = [d_u[s] / lz for s in keep]
        idx = [items.index(s) for s in keep]
        tp = win_probs_np(a[idx])
        thur_pred = (tp / tp.sum()).tolist()

        uz = sum(d_u.values())
        results.append({
            "cell": n, "phrasing": ph, "model": m, "deleted": out,
            "deleted_p": d_u[out] / uz, "keep": keep, "actual": actual,
            "luce": luce_pred, "thurstone": thur_pred,
            "rmse_luce": rmse(luce_pred, actual),
            "rmse_thurstone": rmse(thur_pred, actual),
            "H_unq": entropy_norm([d_u[s] / uz for s in d_u]),
        })

    (HERE / "perm_results.json").write_text(json.dumps(results, indent=1))
    print(f"{len(results)} usable cells")

    def report(name, ss):
        if not ss:
            return
        tw = sum(r["rmse_thurstone"] < r["rmse_luce"] for r in ss)
        from math import comb
        nn, k = len(ss), sum(r["rmse_thurstone"] < r["rmse_luce"] for r in ss)
        p = sum(comb(nn, i) for i in range(k, nn + 1)) / 2 ** nn
        d = sum(r["rmse_luce"] - r["rmse_thurstone"] for r in ss) / nn
        print(f"{name:<34} n={nn:>3}  Thurstone wins {tw:>3}/{nn}  "
              f"mean diff {d:+.4f}  sign p={p:.4f}")

    report("ALL", results)
    for m in MODELS:
        report(f"  model={m}", [r for r in results if r["model"] == m])
    report("  deleted item was favorite", [r for r in results if r["deleted_p"] >= max(0.5, r["deleted_p"]) and r["deleted"] == max(zip([0], [0]))] if False else [r for r in results if r["deleted_p"] > 0.4])
    report("  deleted minor item (p<0.4)", [r for r in results if r["deleted_p"] <= 0.4])
    report("  non-degenerate prior (H>0.2)", [r for r in results if r["H_unq"] > 0.2])
    report("  degenerate prior (H<=0.2)", [r for r in results if r["H_unq"] <= 0.2])


if __name__ == "__main__":
    main()
