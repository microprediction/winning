"""Transport-mode choice under conditioning, with menu-based deletion.

Menu: {car, bus, train, bike, walk, taxi}. Conditions prepend a context
sentence ("It is raining."). Deletion = omit the item from the menu (no
exclusion phrasing needed). Exact logprobs, 4 menu orders averaged.

Tests:
 1. Luce renormalization vs Thurstone contestant removal for every
    (condition, deleted item, model) cell, scored by KL.
 2. Descriptive: per-condition calibrated location shifts (factor structure:
    do outdoor modes move together under weather conditions?).
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
from exact_restrict import MODELS, key
from exact_analyze import calibrate_np, win_probs_np
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
MODES = ["car", "bus", "train", "bike", "walk", "taxi"]
CONDITIONS = {
    "none": "",
    "rain": "It is raining. ",
    "school-out": "School is out. ",
    "fuel-shortage": "There is a fuel shortage. ",
    "midnight": "It is midnight. ",
    "hurry": "You are in a hurry. ",
    "icy": "The streets are icy. ",
    "sunny": "It is a beautiful spring day. ",
}
N_PERMS = 4


def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "system", "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def menu_dist(options, cond_text, model, rng):
    acc = {o: 0.0 for o in options}
    perms = list(itertools.permutations(options)) if len(options) <= 5 else None
    used = 0
    for k in range(N_PERMS):
        order = list(options)
        rng.shuffle(order)
        menu = ", ".join(order)
        prompt = f"{cond_text}You need to get across town. Pick one of these: {menu}."
        raw = top20(prompt, model)
        slot = {o: 0.0 for o in options}
        for tok, p in raw.items():
            w = tok.strip().lower()
            if w in slot:
                slot[w] += p
        z = sum(slot.values())
        if z > 0.2:
            for o in options:
                acc[o] += slot[o] / z
            used += 1
    if used == 0:
        return None
    return {o: acc[o] / used for o in options}


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-9)) for ai, pi in zip(a, p) if ai > 0)


def run_cell(cond_key, model):
    rng = random.Random(hash((cond_key, model)) & 0xFFFF)
    cond = CONDITIONS[cond_key]
    full = menu_dist(MODES, cond, model, rng)
    if full is None:
        return None
    p_full = [full[m] for m in MODES]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return None

    cells = []
    for out in MODES:
        keep = [m for m in MODES if m != out]
        q = menu_dist(keep, cond, model, rng)
        if q is None:
            continue
        actual = [q[m] for m in keep]
        lz = sum(full[m] for m in keep)
        luce = [full[m] / lz for m in keep]
        idx = [MODES.index(m) for m in keep]
        w = win_probs_np(a[idx])
        thur = (w / w.sum()).tolist()
        cells.append({
            "condition": cond_key, "model": model, "deleted": out,
            "deleted_p": full[out], "keep": keep, "actual": actual,
            "luce": luce, "thurstone": thur,
            "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
        })
    return {"condition": cond_key, "model": model, "full": full,
            "locations": {m: float(x) for m, x in zip(MODES, a)},
            "cells": cells}


def main():
    jobs = [(c, m) for c in CONDITIONS for m in MODELS]
    blocks = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_cell, c, m) for c, m in jobs]
        for fut in as_completed(futs):
            try:
                b = fut.result()
                if b:
                    blocks.append(b)
                    print(f"done {b['condition']} {b['model']}", flush=True)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    (HERE / "transport_results.json").write_text(json.dumps(blocks, indent=1))

    cells = [c for b in blocks for c in b["cells"]]
    n = len(cells)
    diffs = [c["kl_luce"] - c["kl_thur"] for c in cells]
    random.seed(9)
    B = 20000
    boots = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) / n
                   for _ in range(B))
    print(f"\n{n} deletion cells across {len(blocks)} condition-model blocks")
    print(f"mean dKL (Luce-Thurstone) = {sum(diffs)/n:+.4f} "
          f"[{boots[int(0.025*B)]:+.4f}, {boots[int(0.975*B)]:+.4f}]  "
          f"P(Luce better)={sum(b <= 0 for b in boots)/B:.4f}")
    tw = sum(d > 0 for d in diffs)
    print(f"Thurstone wins {tw}/{n} cells")

    # full-menu distributions by condition (gpt-4o) for the factor story
    print("\nfull-menu shares (gpt-4o):")
    hdr = "condition".ljust(14) + "".join(m.rjust(7) for m in MODES)
    print(hdr)
    for b in sorted(blocks, key=lambda b: b["condition"]):
        if b["model"] != "gpt-4o":
            continue
        print(b["condition"].ljust(14)
              + "".join(f"{b['full'][m]:7.2f}" for m in MODES))


if __name__ == "__main__":
    main()
