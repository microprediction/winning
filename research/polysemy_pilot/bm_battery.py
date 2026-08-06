"""Complete choice structure of a language model: all 11 menus over a
4-item universe, exact logprobs, order-randomized.

Analyses per family/model:
  1. Block-Marschak nonnegativity (Falmagne 1978): necessary & sufficient
     for random-utility representability. Violations = machine choice is
     not ANY random utility model.
  2. Fits to the full structure: Luce (3 free params), Thurstone locations
     (3 free params), and best-possible RUM (23-dim ranking simplex).
     Gaps: total KL(actual || fit).
"""
import itertools
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, key
from exact_analyze import win_probs_np
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
N_ORDERS = 6

FAMILIES = {
    "lunch": ["pizza", "sushi", "tacos", "curry"],
    "pet": ["dog", "cat", "parrot", "hamster"],
    "fruit": ["mango", "apple", "banana", "cherry"],
    "music": ["jazz", "rock", "classical", "techno"],
    "vacation": ["beach", "mountains", "city", "desert"],
    "drink": ["coffee", "tea", "juice", "cocoa"],
}


def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "system", "content":
                   "Answer with exactly one of the listed options, verbatim, and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def menu_dist(items, model, rng):
    acc = {x: 0.0 for x in items}
    used = 0
    for _ in range(N_ORDERS):
        order = list(items)
        rng.shuffle(order)
        raw = top20(f"Pick one of these: {', '.join(order)}.", model)
        slot = {x: 0.0 for x in items}
        for tok, p in raw.items():
            w = tok.strip().lower()
            if not w or not w.isalpha():
                continue
            hits = [x for x in items if x.startswith(w)]
            if len(hits) == 1:
                slot[hits[0]] += p
        z = sum(slot.values())
        if z > 0.2:
            for x in items:
                acc[x] += slot[x] / z
            used += 1
    return {x: acc[x] / used for x in items} if used else None


def all_menus(items):
    for r in (2, 3, 4):
        yield from itertools.combinations(items, r)


def fetch_family(fam, items, model):
    rng = random.Random(hash((fam, model)) & 0xFFFF)
    q = {}
    for menu in all_menus(items):
        d = menu_dist(list(menu), model, rng)
        if d is None:
            return None
        q[frozenset(menu)] = d
    return q


def block_marschak(items, q):
    """K(x,S) = sum_{T >= S} (-1)^{|T|-|S|} q(x,T); singleton q(x,{x})=1."""
    U = set(items)
    def qq(x, T):
        if len(T) == 1:
            return 1.0
        return q[frozenset(T)][x]
    Ks = {}
    for x in items:
        for r in range(1, 5):
            for S in itertools.combinations(items, r):
                if x not in S:
                    continue
                S = set(S)
                total = 0.0
                rest = list(U - S)
                for k in range(len(rest) + 1):
                    for extra in itertools.combinations(rest, k):
                        T = S | set(extra)
                        total += (-1) ** k * qq(x, T)
                Ks[(x, tuple(sorted(S)))] = total
    return Ks


def total_kl(q, pred):
    tot = 0.0
    for S, d in q.items():
        for x, a in d.items():
            if a > 0:
                tot += a * math.log(a / max(pred[S][x], 1e-9))
    return tot


def fit_luce(items, q):
    lu = np.zeros(len(items))
    def predict(lu):
        u = np.exp(lu)
        out = {}
        for S in q:
            idx = [items.index(x) for x in S]
            z = u[idx].sum()
            out[S] = {x: float(u[items.index(x)] / z) for x in S}
        return out
    for it in range(400):
        base = total_kl(q, predict(lu))
        g = np.zeros(len(items))
        for i in range(len(items)):
            lu2 = lu.copy(); lu2[i] += 1e-4
            g[i] = (total_kl(q, predict(lu2)) - base) / 1e-4
        lu -= 0.5 * g
        lu -= lu.mean()
    return total_kl(q, predict(lu))


def fit_thurstone(items, q):
    a = np.zeros(len(items))
    def predict(a):
        out = {}
        for S in q:
            idx = [items.index(x) for x in S]
            w = win_probs_np(a[idx])
            out[S] = {x: float(wi) for x, wi in zip([items[i] for i in idx], w)}
        return out
    for it in range(200):
        base = total_kl(q, predict(a))
        g = np.zeros(len(items))
        for i in range(len(items)):
            a2 = a.copy(); a2[i] += 1e-3
            g[i] = (total_kl(q, predict(a2)) - base) / 1e-3
        a -= 0.4 * g
        a -= a.mean()
    return total_kl(q, predict(a))


def fit_rum(items, q):
    """Best RUM: distribution over the 24 rankings, projected gradient."""
    ranks = list(itertools.permutations(items))
    r = np.full(len(ranks), 1.0 / len(ranks))
    chooses = {}
    for S in q:
        for x in S:
            chooses[(S, x)] = np.array(
                [1.0 if min(rk.index(y) for y in S) == rk.index(x) else 0.0
                 for rk in ranks])
    def predict(r):
        return {S: {x: float(max(chooses[(S, x)] @ r, 1e-9)) for x in S} for S in q}
    for it in range(1500):
        pred = predict(r)
        g = np.zeros(len(ranks))
        for S, d in q.items():
            for x, a in d.items():
                if a > 0:
                    g -= a / pred[S][x] * chooses[(S, x)]
        r = r * np.exp(-0.02 * (g - (g @ r)))   # exponentiated gradient
        r = np.clip(r, 1e-12, None)
        r /= r.sum()
    return total_kl(q, predict(r))


def main():
    jobs = [(f, items, m) for f, items in FAMILIES.items() for m in MODELS]
    struct = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_family, f, items, m): (f, m) for f, items, m in jobs}
        for fut in as_completed(futs):
            f, m = futs[fut]
            try:
                qd = fut.result()
                if qd:
                    struct[(f, m)] = qd
                    print(f"fetched {f} {m}", flush=True)
            except Exception as e:
                print(f"ERROR {f} {m}: {e}", file=sys.stderr, flush=True)

    rows = []
    for (f, m), qd in struct.items():
        items = FAMILIES[f]
        Ks = block_marschak(items, qd)
        neg = [(k, v) for k, v in Ks.items() if v < -0.02]
        kl_l = fit_luce(items, qd)
        kl_t = fit_thurstone(items, qd)
        kl_r = fit_rum(items, qd)
        rows.append({
            "family": f, "model": m,
            "bm_violations": len(neg), "bm_total": len(Ks),
            "bm_worst": min(Ks.values()),
            "kl_luce_fit": kl_l, "kl_thurstone_fit": kl_t, "kl_best_rum": kl_r,
            "q": {" ".join(sorted(S)): d for S, d in qd.items()},
            "neg_terms": [[list(k[1]), k[0], round(v, 4)] for k, v in neg],
        })
        print(f"{f:<10} {m:<12} BM viol {len(neg):>2}/{len(Ks)} (worst {min(Ks.values()):+.3f})  "
              f"fit KL: Luce {kl_l:.3f}  Thur {kl_t:.3f}  bestRUM {kl_r:.3f}", flush=True)

    (HERE / "bm_results.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    print(f"\nsummary over {n} family-model structures:")
    print(f"  structures with BM violations: {sum(r['bm_violations'] > 0 for r in rows)}/{n}")
    print(f"  total fit KL:  Luce {sum(r['kl_luce_fit'] for r in rows):.2f}   "
          f"Thurstone {sum(r['kl_thurstone_fit'] for r in rows):.2f}   "
          f"best RUM {sum(r['kl_best_rum'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
