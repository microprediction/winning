"""The machine battery on the human dataset's own items, for a matched comparison.

The human odds shift comes from Kamishima's SUSHI rankings; the machine shift so
far comes from colours, metals and the rest. Different domains and different
populations, so the ratio between them is an order-of-magnitude claim rather than
a measurement. This runs the identical pair-restriction statistic on the identical
ten items, under preference framing on both sides, so the two numbers become
comparable.

    delta_ij = log(q_i/q_j) - log(p_i/p_j)

p from the unrestricted preference question, q from the choice named down to the
pair, listing order crossed because order alone moves the statistic by more than
either choice model predicts. Luce requires zero.

Usage:  python sushi_machine.py
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
from models import HEADLINE, BREADTH
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "sushi_machine_raw.jsonl")
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]

# The ten items of sushi3a.5000.10.order, in dataset order.
ITEMS = ["ebi", "anago", "maguro", "ika", "uni", "sake", "tamago", "toro",
         "tekka", "kappa"]
GLOSS = {"ebi": "shrimp", "anago": "sea eel", "maguro": "tuna",
         "ika": "squid", "uni": "sea urchin", "sake": "salmon roe",
         "tamago": "egg", "toro": "fatty tuna", "tekka": "tuna roll",
         "kappa": "cucumber roll"}


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


MENU = ", ".join(f"{i} ({GLOSS[i]})" for i in ITEMS)

# The human p is a share of 5,000 respondents, not one person's token
# distribution, so the machine analogue must also be a population quantity. Each
# voter is a persona-plus-phrasing occasion answering decisively, and the vote
# share across occasions is the machine's p, matching the human construction.
VOTERS = [
    "", "You grew up in Tokyo. ", "You rarely eat fish. ",
    "You are a chef. ", "You dislike strong flavours. ",
    "You love rich, fatty food. ", "You are health-conscious. ",
    "You are a child. ", "You are adventurous with food. ",
    "You prefer simple things. ", "You eat sushi weekly. ",
    "You are on a budget. ", "You have refined, expensive taste. ",
]
ASKS = ["Which sushi do you like best",
        "Which sushi would you order",
        "Which sushi is your favourite"]


def _vote(pr, model, items):
    """One occasion's decisive answer, or None if it names nothing known."""
    m = mass(ask(pr, model), items)
    z = sum(m.values())
    if z <= 0:
        return None
    best = max(m, key=m.get)
    return best if (z > 0.4 and m[best] / z > 0.5) else None


def full_dist(model):
    """Vote share over occasions: the analogue of the human population share."""
    tally = {i: 0 for i in ITEMS}
    for v in VOTERS:
        for a in ASKS:
            pr = (f"{v}{a}? Choose one of: {MENU}. "
                  f"Answer with the single word naming your choice.")
            w = _vote(pr, model, ITEMS)
            if w:
                tally[w] += 1
    n = sum(tally.values())
    return {k: c / n for k, c in tally.items()} if n >= 10 else None


def pair_logodds(i, j, model):
    """Vote share restricted to the pair, listing order crossed per occasion."""
    tally = {i: 0, j: 0}
    for v in VOTERS:
        for a in ASKS:
            for x, y in ((i, j), (j, i)):
                pr = (f"{v}{a}, {x} ({GLOSS[x]}) or {y} ({GLOSS[y]})? "
                      f"Answer with the single word naming your choice.")
                w = _vote(pr, model, [i, j])
                if w:
                    tally[w] += 1
    n = tally[i] + tally[j]
    if n < 8 or min(tally.values()) == 0:
        return None
    return math.log(tally[i] / tally[j])


def run(model):
    p = full_dist(model)
    if not p:
        return []
    rows = []
    for a in range(len(ITEMS)):
        for b in range(a + 1, len(ITEMS)):
            i, j = ITEMS[a], ITEMS[b]
            if p[i] <= 0 or p[j] <= 0:
                continue
            lo = pair_logodds(i, j, model)
            if lo is None:
                continue
            rows.append({"model": model, "i": i, "j": j,
                         "p_i": p[i], "p_j": p[j],
                         "logodds_full": math.log(p[i] / p[j]),
                         "logodds_pair": lo,
                         "delta": lo - math.log(p[i] / p[j])})
    return rows


def main():
    rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(run, MODELS):
            rows.extend(r)
    write_json_atomic(HERE / "sushi_machine_results.json", rows)
    if not rows:
        print("no usable pairs")
        return
    d = [r["delta"] for r in rows]
    n = len(d)
    random.seed(4)
    B = 20000
    bs = sorted(sum(random.choice(d) for _ in range(n)) / n for _ in range(B))
    print(f"MACHINE on the sushi items: {n} pairs over "
          f"{len({r['model'] for r in rows})} models")
    print(f"  mean delta   {sum(d)/n:+.4f} "
          f"[{bs[int(.025*B)]:+.4f}, {bs[int(.975*B)]:+.4f}]")
    print(f"  mean |delta| {sum(abs(x) for x in d)/n:.4f}")
    viol = sum(1 for x in d if abs(x) > 0.1)
    print(f"  |delta| > 0.1 in {viol}/{n} ({100*viol/n:.0f}%)")
    print(f"  shrank toward parity in {sum(1 for x in d if x < 0)}/{n}")
    print()
    print("HUMAN on the same ten items (5,000 rankings, 45 pairs):")
    print("  mean delta   -0.3169 [-0.5581, -0.0804]")
    print("  mean |delta|  0.6941")
    print("  |delta| > 0.1 in 43/45 (96%)")


if __name__ == "__main__":
    main()
