"""Red bus / blue bus: duplicate-alternative test with exact logprobs.

Per family: base menu {bus, train, plane} and split menu {red bus, blue bus,
train, plane}, each averaged over 6 random menu orders. Three zero-parameter
predictions of the aggregated split-menu distribution (bus-total, train,
plane) from the base menu:

  Luce urn:        each duplicate gets the base item's utility ->
                   (2u_b, u_t, u_p) renormalized (bus share inflates)
  indep Thurstone: calibrate base locations, enter TWO bus contestants with
                   the same location (bus share inflates, less)
  substitution:    correlated noise, rho -> 1: shares unchanged

Scored by KL(actual_agg || prediction).
"""
import itertools
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from models import ALL as MODELS, HEADLINE
from exact_analyze import calibrate_np, win_probs_np
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "red_bus_raw.jsonl")  # append-only, before any scoring
N_PERMS = 6

# (family, base item, [variant1, variant2], [other1, other2], token->slot map)
FAMILIES = [
    ("bus",    "bus",    ["red bus", "blue bus"],           ["train", "plane"],
     {"red": 0, "blue": 1, "train": 2, "plane": 3, "bus": 9}),
    ("apple",  "apple",  ["green apple", "red apple"],      ["banana", "orange"],
     {"green": 0, "red": 1, "banana": 2, "orange": 3, "apple": 9}),
    ("cat",    "cat",    ["black cat", "white cat"],        ["dog", "rabbit"],
     {"black": 0, "white": 1, "dog": 2, "rabbit": 3, "cat": 9}),
    ("tea",    "tea",    ["hot tea", "iced tea"],           ["coffee", "juice"],
     {"hot": 0, "iced": 1, "coffee": 2, "juice": 3, "tea": 9}),
    ("wine",   "wine",   ["red wine", "white wine"],        ["beer", "cider"],
     {"red": 0, "white": 1, "beer": 2, "cider": 3, "wine": 9}),
    ("guitar", "guitar", ["acoustic guitar", "electric guitar"], ["piano", "drums"],
     {"acoustic": 0, "electric": 1, "piano": 2, "drums": 3, "guitar": 9}),
    ("rice",   "rice",   ["brown rice", "white rice"],      ["pasta", "bread"],
     {"brown": 0, "white": 1, "pasta": 2, "bread": 3, "rice": 9}),
    ("car",    "car",    ["electric car", "diesel car"],    ["bike", "bus"],
     {"electric": 0, "diesel": 1, "bike": 2, "bus": 3, "car": 9}),
    ("rose",   "rose",   ["red rose", "white rose"],        ["tulip", "daisy"],
     {"red": 0, "white": 1, "tulip": 2, "daisy": 3, "rose": 9}),
    ("bread",  "bread",  ["white bread", "brown bread"],    ["bagel", "croissant"],
     {"white": 0, "brown": 1, "bagel": 2, "croissant": 3, "bread": 9}),
]


def _top20_api(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "system", "content": "Answer with exactly one of the listed options, verbatim, and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def top20(prompt, model):
    """Logged fetch: every response lands on disk before scoring."""
    return RAW.fetch(model, prompt, lambda: _top20_api(prompt, model))


def menu_dist(options, model, tokmap, nslots, rng):
    """Average first-token distribution over N_PERMS menu orders."""
    acc = [0.0] * nslots
    perms = list(itertools.permutations(options))
    rng.shuffle(perms)
    used = 0
    for perm in perms[:N_PERMS]:
        menu = ", ".join(perm)
        raw = top20(f"Pick one of these at random: {menu}.", model)
        slot_p = [0.0] * nslots
        for tok, p in raw.items():
            w = tok.strip().lower()
            if w in tokmap and tokmap[w] < nslots:
                slot_p[tokmap[w]] += p
        z = sum(slot_p)
        if z > 0.2:  # discard degenerate reads
            acc = [a + s / z for a, s in zip(acc, slot_p)]
            used += 1
    if used == 0:
        return None
    return [a / used for a in acc]


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-9)) for ai, pi in zip(a, p) if ai > 0)


def run_family(fam, model, seed):
    name, base, variants, others, tokmap = fam
    rng = random.Random(seed)
    base_map = {base.split()[0]: 0, others[0]: 1, others[1]: 2}
    p3 = menu_dist([base] + others, model, base_map, 3, rng)
    vm = {k: v for k, v in tokmap.items() if v < 4}
    q4 = menu_dist(variants + others, model, vm, 4, rng)
    if p3 is None or q4 is None:
        return None

    actual_agg = [q4[0] + q4[1], q4[2], q4[3]]

    # Luce urn: duplicate the base utility
    u = p3
    luce = [2 * u[0], u[1], u[2]]
    luce = [x / sum(luce) for x in luce]

    # independent Thurstone: two contestants at the base location
    a, err = calibrate_np(p3)
    if err > 0.05:
        return None
    w = win_probs_np([a[0], a[0], a[1], a[2]])
    thur = [float(w[0] + w[1]), float(w[2]), float(w[3])]

    subst = list(p3)  # correlated noise: shares unchanged

    return {
        "family": name, "model": model,
        "base3": p3, "split4": q4, "actual_agg": actual_agg,
        "kl_luce": kl(actual_agg, luce), "kl_indep_thur": kl(actual_agg, thur),
        "kl_substitution": kl(actual_agg, subst),
        "bus_share_base": p3[0], "bus_share_split": actual_agg[0],
        "pred_share_luce": luce[0], "pred_share_thur": thur[0],
    }


def main():
    jobs = [(f, m, i) for i, f in enumerate(FAMILIES) for m in MODELS]
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_family, f, m, 1000 + s) for (f, m, s) in jobs]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)

    write_json_atomic(HERE / "red_bus_results.json", results)
    n = len(results)
    print(f"{n} family-model cells")
    print(f"{'':<24} duplicate-share: base -> split (pred Luce / pred iThur)")
    for r in sorted(results, key=lambda r: (r['family'], r['model'])):
        print(f"{r['family']:<8} {r['model']:<12} "
              f"{r['bus_share_base']:.2f} -> {r['bus_share_split']:.2f}  "
              f"(L {r['pred_share_luce']:.2f} / T {r['pred_share_thur']:.2f})")

    for key_ in ["kl_luce", "kl_indep_thur", "kl_substitution"]:
        tot = sum(r[key_] for r in results)
        wins = sum(1 for r in results
                   if r[key_] == min(r["kl_luce"], r["kl_indep_thur"], r["kl_substitution"]))
        print(f"{key_:<18} total KL={tot:8.3f}   best in {wins}/{n} cells")


if __name__ == "__main__":
    main()
