"""Is the model a hidden electorate?

A mixture of Luce choosers is not a Luce chooser: mixed logit violates IIA and,
by McFadden and Train, can approximate any random utility model. So one
explanation of the odds failure in Section 4 needs no performance noise at all.
If a next-token distribution is effectively a mixture over latent contexts, the
personas the model might adopt, the readings it might take, then the model is
already an electorate and its choice is already non-Lucean for the same reason a
committee of heterogeneous Luce voters is.

That hypothesis makes a directional prediction the alternative does not.
Conditioning on a persona collapses the mixture toward one component, so the
conditioned choice should be closer to Luce than the unconditioned one, and the
odds shift under restriction should shrink. A Gaussian-performance-noise account
predicts nothing of the kind: noise attaches to the item, not to the identity of
the chooser, so conditioning should leave the violation intact.

The measurement is the pair-restriction statistic of Section 4, run twice on the
same categories and pairs:

    delta = log(q_i/q_j) - log(p_i/p_j)

unconditioned, and then averaged over K persona conditions. Listing order is
crossed in both arms, since order alone moves the same statistic by more than
either hypothesis predicts.

Usage:  python mixture_test.py [n_categories] [n_personas]
"""
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from inventory import INVENTORY
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "mixture_raw.jsonl")
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]

# Personas that plausibly shift preference without naming any item.
PERSONAS = [
    "You are a cautious, traditional person who likes familiar things.",
    "You are adventurous and drawn to the unusual.",
    "You are a practical engineer who values function over beauty.",
    "You are an artist who values beauty over function.",
    "You grew up in a small rural town.",
    "You have lived in large cities all your life.",
]


def _api(pr, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0,
        messages=[{"role": "system",
                   "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": pr}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def ask(pr, model):
    return RAW.fetch(model, pr, lambda: _api(pr, model))


def mass(raw, items):
    agg = {i: 0.0 for i in items}
    for tok, p in raw.items():
        w = tok.strip().lower()
        if w in agg:
            agg[w] += p
    return agg


def delta_for(cat, model, prefix, i, j, full_odds):
    """Order-averaged shift in the i:j log odds when the set is named down to
    the pair. Returns None if either order fails to put mass on both items."""
    los = []
    for a, b in ((i, j), (j, i)):
        pr = (f"{prefix}Name a random {cat}, choosing only between "
              f"{a} and {b}.")
        m = mass(ask(pr, model), [i, j])
        if min(m.values()) <= 0:
            return None
        los.append(math.log(m[i] / m[j]))
    return sum(los) / len(los) - full_odds


def run(cat, model, n_personas):
    inv = INVENTORY[cat]

    def field(prefix):
        d = mass(ask(f"{prefix}Name a random {cat}.", model), inv)
        live = [k for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if v > 1e-4]
        if len(live) < 3:
            return None, None
        z = sum(d[k] for k in live)
        return live, {k: d[k] / z for k in live}

    live, p = field("")
    if not live:
        return []
    pairs = [(live[0], live[1])]
    if len(live) > 2:
        pairs.append((live[0], live[2]))
    if len(live) > 3:
        pairs.append((live[1], live[3]))

    rows = []
    for i, j in pairs:
        base = math.log(p[i] / p[j])
        d_un = delta_for(cat, model, "", i, j, base)
        if d_un is None:
            continue
        # conditioned arm: each persona gets its own unrestricted odds, because
        # a persona changes the field as well as the restriction
        ds = []
        for k in range(n_personas):
            pre = PERSONAS[k % len(PERSONAS)] + " "
            lv, pp = field(pre)
            if not lv or i not in pp or j not in pp:
                continue
            d = delta_for(cat, model, pre, i, j, math.log(pp[i] / pp[j]))
            if d is not None:
                ds.append(d)
        if len(ds) < max(2, n_personas // 2):
            continue
        rows.append({"category": cat, "model": model, "i": i, "j": j,
                     "delta_unconditioned": d_un,
                     "delta_conditioned_mean": sum(ds) / len(ds),
                     "abs_unconditioned": abs(d_un),
                     "abs_conditioned_mean": sum(abs(x) for x in ds) / len(ds),
                     "n_personas_used": len(ds)})
    return rows


def main():
    ncat = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    npers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cats = [c for c in INVENTORY if len(INVENTORY[c]) >= 6][:ncat]
    jobs = [(c, m) for c in cats for m in MODELS]
    print(f"{len(cats)} categories x {len(MODELS)} models x {npers} personas, "
          f"{len(RAW)} cached", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run, c, m, npers) for c, m in jobs]
        for k, f in enumerate(as_completed(futs)):
            try:
                rows.extend(f.result())
            except Exception as e:
                print(f"ERROR {str(e)[:90]}", file=sys.stderr, flush=True)
            if (k + 1) % 6 == 0:
                print(f"  {k+1}/{len(jobs)}", flush=True)
    write_json_atomic(HERE / "mixture_results.json", rows)
    if not rows:
        print("no usable pairs")
        return

    def clus(vals, cats_):
        g = defaultdict(list)
        for v, c in zip(vals, cats_):
            g[c].append(v)
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

    cs = [r["category"] for r in rows]
    print(f"\n{len(rows)} pairs over {len(set(cs))} categories")
    for lab, k in (("unconditioned |delta|", "abs_unconditioned"),
                   ("persona-conditioned |delta|", "abs_conditioned_mean")):
        m, lo, hi = clus([r[k] for r in rows], cs)
        print(f"  {lab:<30} {m:.4f} [{lo:.4f}, {hi:.4f}]")
    diff = [r["abs_unconditioned"] - r["abs_conditioned_mean"] for r in rows]
    m, lo, hi = clus(diff, cs)
    print(f"  reduction from conditioning     {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("  positive means conditioning moves the model TOWARD Luce,")
    print("  which is what the hidden-electorate hypothesis predicts.")
    print(f"  smaller in {sum(1 for d in diff if d > 0)}/{len(diff)} pairs")


if __name__ == "__main__":
    main()
