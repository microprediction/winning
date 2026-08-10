"""The canonical IIA test: does an odds ratio survive removing everything else?

Every battery so far scores a whole restricted distribution, which entangles
three things: where mass goes, how spread out it is, and how large the surviving
field is. The classical statement of Luce's axiom entangles none of them. For any
two items,

    P(i) / P(j)  must not depend on what else is available,

so restricting the choice set to exactly {i, j} is the sharpest possible probe.
The statistic is a log odds ratio,

    delta_ij = log[ q_i / q_j ]  -  log[ p_i / p_j ],

which is invariant to any rescaling of either distribution. No entropy
correction, no field-size effect, no tilt parameter can enter: Luce predicts
delta = 0 exactly, and a contest predicts delta shrunk toward zero by an amount
computable from the locations, because removing the rest of the field changes
how hard each survivor has to win.

Three conditions per pair, to separate the axiom from the arithmetic:

  full     the unrestricted prompt, giving p_i / p_j
  pair     the choice named down to exactly i and j
  triple   i, j and one distractor k, so the same odds are measured with one
           extra competitor present

Under IIA the odds are identical in all three. Under a contest they move in a
signed, predictable direction. Because delta is scale-free, this design cannot
be rescued or defeated by the flattening that dominates the open-vocabulary
batteries.

Usage:  python binary_iia.py [n_categories]
"""
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from exact_analyze import calibrate_np, win_probs_np
from inventory import INVENTORY
from models import HEADLINE, BREADTH
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "binary_iia_raw.jsonl")
MODELS = HEADLINE + BREADTH
N_PAIRS = 4          # pairs per category, spanning the odds range
STOP = set("the and for a an of to in on it is".split())


def _api(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0,
        messages=[{"role": "system",
                   "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def top20(prompt, model):
    return RAW.fetch(model, prompt, lambda: _api(prompt, model))


def mass(raw, items):
    agg = {i: 0.0 for i in items}
    for tok, p in raw.items():
        w = tok.strip().lower()
        if w in agg:
            agg[w] += p
    return agg


def run_category(cat, model):
    inv = INVENTORY[cat]
    full = top20(f"Name a random {cat}.", model)
    d = mass(full, inv)
    live = [k for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v > 1e-4]
    if len(live) < 4:
        return []
    z = sum(d[k] for k in live)
    p = {k: d[k] / z for k in live}

    # pairs spanning the odds range: adjacent, near, far
    pairs = []
    for gap in (1, 2, 3):
        for a in range(min(2, len(live) - gap)):
            b = a + gap
            if b < len(live):
                pairs.append((live[a], live[b]))
    pairs = pairs[:N_PAIRS]
    distractor = live[-1]

    rows = []
    for i, j in pairs:
        if distractor in (i, j):
            continue
        pair_raw = top20(f"Name a random {cat}, choosing only between "
                         f"{i} and {j}.", model)
        # order control: the same pair with the listing reversed
        swap_raw = top20(f"Name a random {cat}, choosing only between "
                         f"{j} and {i}.", model)
        qs = mass(swap_raw, [i, j])
        trip_raw = top20(f"Name a random {cat}, choosing only between "
                         f"{i}, {j} and {distractor}.", model)
        qp = mass(pair_raw, [i, j])
        qt = mass(trip_raw, [i, j, distractor])
        if min(qp.values()) <= 0 or qt[i] <= 0 or qt[j] <= 0:
            continue
        lo_full = math.log(p[i] / p[j])
        lo_pair = math.log(qp[i] / qp[j])
        lo_trip = math.log(qt[i] / qt[j])

        # contest prediction for the pair, from locations fitted to p
        items = live
        a_loc, err = calibrate_np([p[k] for k in items])
        if err > 0.05:
            continue
        ii, jj = items.index(i), items.index(j)
        w = win_probs_np(a_loc[[ii, jj]])
        lo_thur = math.log(w[0] / w[1]) if w[1] > 0 else float("nan")

        lo_swap = (math.log(qs[i] / qs[j])
                   if min(qs.values()) > 0 else float("nan"))
        rows.append({"logodds_swap": lo_swap,
                     "delta_swap": lo_swap - lo_full,
                     "category": cat, "model": model, "i": i, "j": j,
                     "p_i": p[i], "p_j": p[j],
                     "logodds_full": lo_full, "logodds_pair": lo_pair,
                     "logodds_triple": lo_trip,
                     "logodds_thurstone_pair": lo_thur,
                     "delta_pair": lo_pair - lo_full,
                     "delta_triple": lo_trip - lo_full,
                     "delta_thurstone": lo_thur - lo_full,
                     "field": len(items)})
    return rows


def main():
    ncat = int(sys.argv[1]) if len(sys.argv) > 1 else len(INVENTORY)
    cats = list(INVENTORY)[:ncat]
    jobs = [(c, m) for c in cats for m in MODELS]
    print(f"{len(jobs)} category-model units, {len(RAW)} responses cached",
          flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run_category, c, m) for c, m in jobs]
        for k, f in enumerate(as_completed(futs)):
            try:
                rows.extend(f.result())
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 25 == 0:
                print(f"{k+1}/{len(jobs)}", flush=True)
    write_json_atomic(HERE / "binary_iia_results.json", rows)
    print(f"\n{len(rows)} pair cells")
    if not rows:
        return

    def stat(vals, groups):
        g = {}
        for v, c in zip(vals, groups):
            g.setdefault(c, []).append(v)
        keys = list(g)
        random.seed(4)
        B = 20000
        means = []
        for _ in range(B):
            pick = [g[keys[random.randrange(len(keys))]] for _ in keys]
            flat = [x for gg in pick for x in gg]
            means.append(sum(flat) / len(flat))
        means.sort()
        flat = [x for gg in g.values() for x in gg]
        return sum(flat) / len(flat), means[int(.025 * B)], means[int(.975 * B)]

    cats_g = [r["category"] for r in rows]
    ok = [r for r in rows if r["delta_swap"] == r["delta_swap"]]
    if ok:
        m, lo, hi = stat([r["delta_swap"] for r in ok],
                         [r["category"] for r in ok])
        print(f"  {'observed shift, order swapped':<28} mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        avg = [(r["delta_pair"] + r["delta_swap"]) / 2 for r in ok]
        m, lo, hi = stat(avg, [r["category"] for r in ok])
        print(f"  {'order-averaged shift':<28} mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              "   <-- position bias cancels here")
    for label, k in (("observed shift, pair", "delta_pair"),
                     ("observed shift, triple", "delta_triple"),
                     ("contest prediction, pair", "delta_thurstone")):
        m, lo, hi = stat([r[k] for r in rows], cats_g)
        print(f"  {label:<28} mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("  Luce predicts exactly 0.0000 for all three")
    shrink = [r for r in rows if abs(r["logodds_pair"]) < abs(r["logodds_full"])]
    print(f"  odds moved toward parity in {len(shrink)}/{len(rows)} pairs")
    signdis = [r for r in rows
               if r["logodds_pair"] * r["logodds_full"] < 0]
    print(f"  odds REVERSED in {len(signdis)}/{len(rows)} pairs")


if __name__ == "__main__":
    main()
