"""Decoy (asymmetric dominance) and compromise effects with exact logprobs.

Both violate REGULARITY -- P(x | S) can never increase when S grows -- which
every random utility model satisfies (Luce and Thurstone alike). Human
choosers violate it (Huber, Payne & Puto 1982; Simonson 1989).

Design per family/model/test:
  2-menu {T, C}          -> P2(T)
  3-menu {T, C, D}       -> P3(T), decoy D dominated by T (decoy test)
                            or extreme E making M the compromise
Options are lettered; letters are randomized over options across 6 draws and
the letter-token logprobs read exactly. Attraction/compromise effect size =
relative share change; regularity violation = absolute share increase.
"""
import json
import math
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from models import ALL as MODELS, HEADLINE
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "context_raw.jsonl")  # append-only, before any scoring


def _top20_api(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0, messages=[{"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}
N_DRAWS = 6

# family: product, attr names, T(arget), C(ompetitor), D(ecoy dominated by T),
# and for compromise: E1, M, E2 (M mid on both; E1/E2 extremes)
FAMILIES = [
    {"product": "laptop", "attrs": ("price", "battery life"),
     "T": ("$900", "12 hours"), "C": ("$600", "6 hours"), "D": ("$950", "11 hours"),
     "E1": ("$500", "5 hours"), "M": ("$800", "9 hours"), "E2": ("$1100", "13 hours")},
    {"product": "apartment", "attrs": ("monthly rent", "commute time"),
     "T": ("$1800", "15 minutes"), "C": ("$1200", "45 minutes"), "D": ("$1900", "18 minutes"),
     "E1": ("$1000", "55 minutes"), "M": ("$1500", "30 minutes"), "E2": ("$2100", "10 minutes")},
    {"product": "coffee maker", "attrs": ("price", "customer rating"),
     "T": ("$120", "4.7 stars"), "C": ("$60", "3.9 stars"), "D": ("$130", "4.6 stars"),
     "E1": ("$40", "3.5 stars"), "M": ("$85", "4.2 stars"), "E2": ("$160", "4.8 stars")},
    {"product": "flight", "attrs": ("price", "total travel time"),
     "T": ("$450", "6 hours"), "C": ("$250", "13 hours"), "D": ("$470", "7 hours"),
     "E1": ("$200", "16 hours"), "M": ("$350", "9 hours"), "E2": ("$550", "5 hours")},
    {"product": "pair of headphones", "attrs": ("price", "sound quality rating"),
     "T": ("$250", "9.2/10"), "C": ("$120", "7.4/10"), "D": ("$270", "9.0/10"),
     "E1": ("$80", "6.5/10"), "M": ("$180", "8.3/10"), "E2": ("$350", "9.6/10")},
    {"product": "phone plan", "attrs": ("monthly cost", "data allowance"),
     "T": ("$55", "30 GB"), "C": ("$30", "8 GB"), "D": ("$60", "28 GB"),
     "E1": ("$20", "3 GB"), "M": ("$40", "15 GB"), "E2": ("$70", "50 GB")},
]

LETTERS = list(string.ascii_uppercase[:6])


def letter_dist(options, product, attrs, model, rng):
    """options: list of (tag, (v1, v2)). Returns mean prob per tag."""
    acc = {tag: 0.0 for tag, _ in options}
    used = 0
    for _ in range(N_DRAWS):
        opts = list(options)
        rng.shuffle(opts)
        letters = LETTERS[:len(opts)]
        lines = [f"{L}: {v1} {attrs[0]}, {v2} {attrs[1]}"
                 for L, (tag, (v1, v2)) in zip(letters, opts)]
        prompt = (f"You are choosing a {product}. Options:\n"
                  + "\n".join(lines)
                  + "\nWhich do you choose? Answer with the letter only.")
        raw = RAW.fetch(model, prompt, lambda: _top20_api(prompt, model))
        slot = {tag: 0.0 for tag, _ in opts}
        for tok, p in raw.items():
            w = tok.strip().upper()
            for L, (tag, _) in zip(letters, opts):
                if w == L:
                    slot[tag] += p
        z = sum(slot.values())
        if z > 0.2:
            for tag in slot:
                acc[tag] += slot[tag] / z
            used += 1
    if used == 0:
        return None
    return {tag: v / used for tag, v in acc.items()}


def run_family(fam, model, seed):
    rng = random.Random(seed)
    at = fam["attrs"]
    out = {"product": fam["product"], "model": model}

    # decoy test
    p2 = letter_dist([("T", fam["T"]), ("C", fam["C"])], fam["product"], at, model, rng)
    p3 = letter_dist([("T", fam["T"]), ("C", fam["C"]), ("D", fam["D"])],
                     fam["product"], at, model, rng)
    if p2 and p3:
        rel3 = p3["T"] / (p3["T"] + p3["C"])
        out["decoy"] = {"P2_T": p2["T"], "P3_T_abs": p3["T"], "P3_D": p3["D"],
                        "rel_T_3": rel3, "attraction": rel3 - p2["T"],
                        "regularity_violated": p3["T"] > p2["T"]}

    # compromise test
    q2 = letter_dist([("E1", fam["E1"]), ("M", fam["M"])], fam["product"], at, model, rng)
    q3 = letter_dist([("E1", fam["E1"]), ("M", fam["M"]), ("E2", fam["E2"])],
                     fam["product"], at, model, rng)
    if q2 and q3:
        rel3 = q3["M"] / (q3["M"] + q3["E1"])
        out["compromise"] = {"P2_M": q2["M"], "P3_M_abs": q3["M"], "P3_E2": q3["E2"],
                             "rel_M_3": rel3, "compromise": rel3 - q2["M"],
                             "regularity_violated": q3["M"] > q2["M"]}
    return out


def main():
    jobs = [(f, m, 100 + i) for i, f in enumerate(FAMILIES) for m in MODELS]
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_family, f, m, s) for f, m, s in jobs]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    write_json_atomic(HERE / "context_effects_results.json", results)

    for test in ["decoy", "compromise"]:
        rs = [r[test] for r in results if test in r]
        if not rs:
            continue
        eff = [r["attraction" if test == "decoy" else "compromise"] for r in rs]
        viol = sum(r["regularity_violated"] for r in rs)
        mean = sum(eff) / len(eff)
        eff.sort()
        print(f"{test}: n={len(rs)}  mean relative-share effect {mean:+.4f}  "
              f"median {eff[len(eff)//2]:+.4f}  absolute regularity violations {viol}/{len(rs)}")
    for r in sorted(results, key=lambda r: (r['product'], r['model'])):
        d = r.get("decoy", {})
        c = r.get("compromise", {})
        print(f"{r['product']:<20} {r['model']:<12} "
              f"decoy: {d.get('P2_T', float('nan')):.2f}->{d.get('rel_T_3', float('nan')):.2f}"
              f"{' VIOL' if d.get('regularity_violated') else '     '} "
              f"compromise: {c.get('P2_M', float('nan')):.2f}->{c.get('rel_M_3', float('nan')):.2f}"
              f"{' VIOL' if c.get('regularity_violated') else ''}")


if __name__ == "__main__":
    main()
