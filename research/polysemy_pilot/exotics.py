"""Exotics: ordered selection to arbitrary depth, exact log probabilities.

The two-slot battery is an exacta. Extend it to a trifecta, a superfecta, and
on to slot five, and the comparison stops being a single number and becomes a
curve. Each slot is elicited separately with the earlier picks fed back in, so
every conditional distribution is measured exactly rather than inferred:

  First: ___
  First: gold, Second: ___
  First: gold, Second: silver, Third: ___
  ... to DEPTH slots

Plackett-Luce / Harville predicts slot d by renormalizing the slot-1
distribution over the items not yet taken. Thurstone removes the taken
contestants from the field and recomputes winning probabilities. Harville's
known failure on racing exotics is an overpricing of the favorite in later
positions, and it compounds with depth: at slot d there are d-1 removals, and
renormalization has no way to redistribute the vacated mass except in
proportion. The prediction is therefore a monotone depth curve, with the Luce
deficit widening from slot 2 to slot DEPTH.

Because each category and model contributes the whole ladder, the widening is
measured within category rather than across designs, which removes the worry
that deeper slots simply draw on easier categories.

Every response is appended to exotics_raw.jsonl before anything is scored.
"""
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from inventory import INVENTORY
from models import ALL as MODELS, HEADLINE
from datastore import append_jsonl, load_jsonl, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW_LOG = HERE / "exotics_raw.jsonl"
DEPTH = 5
SLOTS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
NUMWORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}

# Elicitation architectures. `dir` marks the two preference-ordered frames
# that form the reversibility pair: under a Thurstonian scale they are one
# latent vector read in two directions, since the minimum of X is the maximum
# of -X. Plackett-Luce must instead invert its utilities, and forward and
# reverse Plackett-Luce are not the same distribution over rankings.
FRAMES = [
    {"name": "random", "dir": None,
     "t": "Select {n} random {p} in order."},
    {"name": "different", "dir": None,
     "t": "Name {n} different {p}, in order."},
    {"name": "best-first", "dir": "best",
     "t": "Rank {n} {p} from most favourite to least favourite."},
    {"name": "worst-first", "dir": "worst",
     "t": "Rank {n} {p} from least favourite to most favourite."},
]


def plural(cat):
    """Pluralize the head noun: first word for 'X of/in Y', else last word."""
    words = cat.split()
    i = 0 if (" of " in f" {cat} " or " in " in f" {cat} ") else len(words) - 1
    w = words[i]
    if w.endswith(("s", "x", "z", "ch", "sh")):
        w += "es"
    elif len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
        w = w[:-1] + "ies"
    else:
        w += "s"
    return " ".join(words[:i] + [w] + words[i + 1:])


def ask(frame, cat, model, taken):
    """Measure the next slot given the items already taken, in order."""
    filled = ", ".join(f"{SLOTS[i]}: {t}" for i, t in enumerate(taken))
    blank = f"{SLOTS[len(taken)]}: ___"
    inner = f"{filled}, {blank}" if filled else blank
    prompt = (f"{frame['t'].format(n=NUMWORD[DEPTH], p=plural(cat))} "
              f'Fill in the blank with a single word: "{inner}" '
              "Give only the missing word.")
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0, messages=[{"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def inv_dist(raw, cat):
    inv = set(INVENTORY[cat])
    out = {}
    for tok, p in raw.items():
        w = tok.strip().lower()
        if w in inv:
            out[w] = out.get(w, 0.0) + p
    return out


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def fetch(cat, fi, model, cache):
    """Walk the ladder, appending each response the moment it arrives. Stops
    early when the measured field runs out of candidates."""
    ck = f"{cat}||{fi}||{model}"
    if ck in cache:
        return cache[ck]
    frame = FRAMES[fi]
    rec = {"key": ck, "category": cat, "frame": fi, "model": model, "taken": []}
    try:
        rec["s1"] = ask(frame, cat, model, [])
        field = inv_dist(rec["s1"], cat)
        taken = []
        while len(taken) < DEPTH - 1:
            remaining = {k: v for k, v in field.items() if k not in taken}
            if len(remaining) < 2:
                break  # nothing left to predict
            taken.append(max(remaining, key=remaining.get))
            raw = ask(frame, cat, model, taken)
            rec[f"s{len(taken) + 1}"] = raw
            # later picks come from what the model actually offers at that slot
            seen = inv_dist(raw, cat)
            for k, v in seen.items():
                field.setdefault(k, 0.0)
            rec["taken"] = list(taken)
    finally:
        append_jsonl(RAW_LOG, rec)  # keep whatever was paid for, even on error
        cache[ck] = rec
    return rec


def score(rec):
    """Score every available slot against both families. Pure: no API."""
    cat = rec["category"]
    if "s1" not in rec:
        return []
    d1 = inv_dist(rec["s1"], cat)
    if len(d1) < 3:
        return []
    items = sorted(d1, key=d1.get, reverse=True)
    z1 = sum(d1.values())
    p_full = [d1[s] / z1 for s in items]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return []

    out = []
    for depth in range(2, DEPTH + 1):
        raw = rec.get(f"s{depth}")
        if raw is None:
            break
        taken = rec["taken"][:depth - 1]
        dq = inv_dist(raw, cat)
        keep = [s for s in items if s not in taken and s in dq]
        if len(keep) < 2:
            break
        az = sum(dq[s] for s in keep)
        actual = [dq[s] / az for s in keep]
        lz = sum(d1[s] for s in keep)
        luce = [d1[s] / lz for s in keep]  # Plackett-Luce / Harville
        idx = [items.index(s) for s in keep]
        w = win_probs_np(a[idx])
        thur = (w / w.sum()).tolist()
        out.append({"category": cat, "frame": rec["frame"], "model": rec["model"],
                    "depth": depth, "n_removed": depth - 1, "taken": taken,
                    "keep": keep, "actual": actual, "luce": luce,
                    "thurstone": thur, "kl_luce": kl(actual, luce),
                    "kl_thur": kl(actual, thur), "H_slot1": entropy_norm(p_full)})
    return out


def reversal(best_rec, worst_rec):
    """Predict the least-favourite-first distribution from the
    most-favourite-first one, zero parameters on both sides.

    Thurstone: calibrate locations to the best-first field, negate them, and
    recompute winning probabilities, since P(X_i minimal) = P(-X_i maximal).
    One latent scale, read backwards.

    Luce: being chosen worst first is being best on inverted utilities, so the
    prediction is proportional to 1/u_i.

    Also returns the model-free disagreement between the two directions, which
    needs no choice model at all: KL of the observed worst-first distribution
    against the observed best-first one.
    """
    cat = best_rec["category"]
    if "s1" not in best_rec or "s1" not in worst_rec:
        return None
    d_b = inv_dist(best_rec["s1"], cat)
    d_w = inv_dist(worst_rec["s1"], cat)
    common = [s for s in sorted(d_b, key=d_b.get, reverse=True) if s in d_w]
    if len(common) < 3:
        return None

    items = sorted(d_b, key=d_b.get, reverse=True)
    zb = sum(d_b.values())
    p_full = [d_b[s] / zb for s in items]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return None

    zw = sum(d_w[s] for s in common)
    actual = [d_w[s] / zw for s in common]

    idx = [items.index(s) for s in common]
    w = win_probs_np(-a[idx])              # same scale, read backwards
    thur = (w / w.sum()).tolist()

    inv = [1.0 / max(d_b[s], 1e-12) for s in common]   # inverted utilities
    zi = sum(inv)
    luce = [v / zi for v in inv]

    zf = sum(d_b[s] for s in common)
    forward = [d_b[s] / zf for s in common]

    return {"category": cat, "model": best_rec["model"], "items": common,
            "actual_worst": actual, "luce": luce, "thurstone": thur,
            "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
            "kl_directions": kl(actual, forward),
            "H_best": entropy_norm(p_full)}


def boot(diffs, seed=4, B=20000):
    n = len(diffs)
    random.seed(seed)
    bs = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) / n
                for _ in range(B))
    return sum(diffs) / n, bs[int(.025 * B)], bs[int(.975 * B)]


def rep(name, ss):
    if not ss:
        print(f"{name:<34} (empty)")
        return
    diffs = [r["kl_luce"] - r["kl_thur"] for r in ss]
    m, lo, hi = boot(diffs)
    print(f"{name:<34} n={len(diffs):>4} wins={sum(d > 0 for d in diffs):>4} "
          f"mean dKL={m:+.4f} [{lo:+.4f},{hi:+.4f}]")


def main():
    cache = load_jsonl(RAW_LOG, key="key")
    jobs = [(c, fi, m) for c in INVENTORY for fi in range(len(FRAMES))
            for m in MODELS]
    fnames = [f["name"] for f in FRAMES]
    todo = [j for j in jobs if f"{j[0]}||{j[1]}||{j[2]}" not in cache]
    print(f"{len(jobs)} ladders ({len(cache)} cached, {len(todo)} to fetch), "
          f"depth {DEPTH}", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(fetch, *j, cache) for j in todo]
        for k, fut in enumerate(as_completed(futs)):
            try:
                fut.result()
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 50 == 0:
                print(f"fetched {k+1}/{len(todo)}", flush=True)

    ladders = [score(cache[f"{c}||{fi}||{m}"]) for c, fi, m in jobs
               if f"{c}||{fi}||{m}" in cache]
    ladders = [l for l in ladders if l]
    results = [r for l in ladders for r in l]
    write_json_atomic(HERE / "exotics_results.json", results)
    print(f"\n{len(ladders)} ladders scored, {len(results)} cells")

    head = [r for r in results if r["model"] in HEADLINE]
    print("\ndepth curve (HEADLINE models)")
    for d in range(2, DEPTH + 1):
        rep(f"  slot {d}", [r for r in head if r["depth"] == d])
    print("\ndepth curve (all models)")
    for d in range(2, DEPTH + 1):
        rep(f"  slot {d}", [r for r in results if r["depth"] == d])
    print("\nby model, deepest available slot")
    for m in MODELS:
        rs = [r for r in results if r["model"] == m]
        if rs:
            dmax = max(r["depth"] for r in rs)
            rep(f"  {m} slot {dmax}", [r for r in rs if r["depth"] == dmax])

    # within-ladder widening: last slot vs slot 2, same category and model
    grow = []
    for l in ladders:
        if len(l) >= 2:
            first, last = l[0], l[-1]
            grow.append((last["kl_luce"] - last["kl_thur"])
                        - (first["kl_luce"] - first["kl_thur"]))
    if grow:
        m, lo, hi = boot(grow, seed=5)
        print(f"\nwithin-ladder widening (deepest - slot 2): n={len(grow)} "
              f"mean={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"positive in {sum(g > 0 for g in grow)}/{len(grow)}")

    # per-frame depth curves: does the verdict hold across question types?
    print("\nby elicitation frame (all depths pooled)")
    for fi, f in enumerate(FRAMES):
        rep(f"  {f['name']}", [r for r in results if r["frame"] == fi])

    # reversibility: predict least-favourite-first from most-favourite-first
    bi, wi = fnames.index("best-first"), fnames.index("worst-first")
    revs = []
    for c in INVENTORY:
        for m in MODELS:
            b = cache.get(f"{c}||{bi}||{m}")
            w = cache.get(f"{c}||{wi}||{m}")
            if b and w:
                r = reversal(b, w)
                if r:
                    revs.append(r)
    write_json_atomic(HERE / "reversibility_results.json", revs)
    print(f"\nreversibility: {len(revs)} category-model pairs")
    rep("  all", revs)
    rep("  HEADLINE", [r for r in revs if r["model"] in HEADLINE])
    rep("  non-degenerate (H>0.2)", [r for r in revs if r["H_best"] > 0.2])
    if revs:
        dirs = [r["kl_directions"] for r in revs]
        print(f"  model-free direction disagreement: mean KL="
              f"{sum(dirs)/len(dirs):.4f}, "
              f"reversed in {sum(d > 0.1 for d in dirs)}/{len(dirs)} pairs")


if __name__ == "__main__":
    main()
