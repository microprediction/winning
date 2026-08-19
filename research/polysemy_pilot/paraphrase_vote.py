"""Preference by majority over many phrasings: real choice that does not collapse.

Asking a preference-tuned model what it likes gives a near-degenerate token
distribution, which is why the restriction batteries had to fall back on "name a
random X" to recover entropy. That fallback buys entropy at the cost of asking
about preference at all.

This design keeps the preference question and moves the randomness somewhere
else. The same question is asked in many phrasings, each answered almost
deterministically, and the distribution of interest is the distribution of
answers across phrasings:

    p_i = fraction of phrasings whose answer is item i.

Every trial is decisive and the aggregate is not, which is exactly the structure
of human choice data: one subject on one occasion picks one thing, and the
distribution lives in the population of occasions. Read Thurstonially, the
phrasing draws the performance noise, so the latent locations belong to the model
and the noise belongs to the elicitation. That is a claim the collapse inside a
single prompt cannot even express.

The restriction is then applied to the identical phrasing set, so p and q are
vote distributions over the same occasions and differ only in the choice set.
Both the KL comparison and the scale-free odds test of Section 4 apply unchanged.

Phrasings are generated combinatorially rather than by a model, so the set is
exactly reproducible and carries no second model's preferences. Openers, personas,
syntactic frames and closers multiply out; none of them mentions or ranks any
item, so the manipulation is orthogonal to the choice.

Usage:  python paraphrase_vote.py [n_phrasings] [n_categories]
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
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from inventory import INVENTORY
from models import HEADLINE
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "paraphrase_raw.jsonl")
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]   # cheap tier

# Components that change wording, register and framing without touching the
# choice set. 6 x 6 x 5 x 4 = 720 combinations, sampled without replacement.
OPENERS = [
    "What is your favourite {c}?",
    "Which {c} do you like best?",
    "If you had to pick one {c}, which would it be?",
    "Name the {c} you prefer above all others.",
    "Of all the options, which {c} appeals to you most?",
    "Tell me which {c} you would choose.",
]
PERSONAS = [
    "",
    "You are a thoughtful person with definite tastes. ",
    "Answer as you genuinely would, not as you think is expected. ",
    "You are being asked casually, by a friend. ",
    "You are filling in a survey. ",
    "Answer spontaneously, without deliberating. ",
]
FRAMES = [
    "Answer with a single word.",
    "Reply with one word only.",
    "Give just the name, nothing else.",
    "One word, please.",
    "Respond with the single word that names your choice.",
]
CLOSERS = [
    "",
    " There is no wrong answer.",
    " Please commit to one.",
    " Do not hedge.",
]


def phrasings(n, seed=11):
    combos = list(itertools.product(range(len(PERSONAS)), range(len(OPENERS)),
                                    range(len(FRAMES)), range(len(CLOSERS))))
    random.Random(seed).shuffle(combos)
    return combos[:n]


def build(combo, cat, excluded=None):
    pi, oi, fi, ci = combo
    q = OPENERS[oi].format(c=cat)
    if excluded:
        q = q.rstrip("?.") + f", other than {excluded}?"
    # The trailing stub forces the answer into first-token position. Without it
    # the model opens with "My" or "I" and the item is read from a low-probability
    # tail, which measures the tail rather than the answer.
    return f"{PERSONAS[pi]}{q} {FRAMES[fi]}{CLOSERS[ci]}\n\nAnswer:"


def _api(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0, messages=[{"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def top20(prompt, model):
    return RAW.fetch(model, prompt, lambda: _api(prompt, model))


MIN_CONFIDENCE = 0.5   # the item must be what the model would actually say


def vote(raw, items):
    """The answer this phrasing gives, or None if the model's answer is not a
    known item. A vote counts only when the item carries at least
    MIN_CONFIDENCE of the first-token mass: otherwise the reading is taken from
    a low-probability tail and measures the tail rather than the answer, which
    manufactures apparent variety in exactly the cells where the model is in
    fact decisive about something we failed to parse."""
    best, bp = None, 0.0
    for tok, p in raw.items():
        w = tok.strip().lower()
        if w in items and p > bp:
            best, bp = w, p
    if bp < MIN_CONFIDENCE:
        return None, bp
    return best, bp


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def run(cat, model, combos):
    inv = set(INVENTORY[cat])
    # unrestricted votes
    tally = {}
    conf = []
    for combo in combos:
        w, pw = vote(top20(build(combo, cat), model), inv)
        if w:
            tally[w] = tally.get(w, 0) + 1
            conf.append(pw)
    n = sum(tally.values())
    if n < len(combos) * 0.5 or len(tally) < 3:
        return None
    items = sorted(tally, key=tally.get, reverse=True)
    p = [tally[i] / n for i in items]

    # restrict: exclude the modal answer, same phrasings
    fav = items[0]
    rt = {}
    for combo in combos:
        w, _ = vote(top20(build(combo, cat, excluded=fav), model), inv)
        if w and w != fav:
            rt[w] = rt.get(w, 0) + 1
    rn = sum(rt.values())
    keep = [i for i in items if i != fav and i in rt]
    if rn < len(combos) * 0.4 or len(keep) < 2:
        return None
    actual = [rt[i] / rn for i in keep]

    lz = sum(p[items.index(i)] for i in keep)
    luce = [p[items.index(i)] / lz for i in keep]
    a, err = calibrate_np(p)
    if err > 0.05:
        return None
    idx = [items.index(i) for i in keep]
    w = win_probs_np(a[idx])
    thur = (w / w.sum()).tolist()

    return {"category": cat, "model": model, "n_phrasings": len(combos),
            "items": items, "votes": [tally[i] for i in items],
            "p": p, "fav": fav, "keep": keep, "actual": actual,
            "luce": luce, "thurstone": thur,
            "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
            "H_votes": entropy_norm(p),
            "mean_within_prompt_confidence": sum(conf) / len(conf)}


def main():
    nph = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ncat = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    combos = phrasings(nph)
    cats = [c for c in INVENTORY if len(INVENTORY[c]) >= 8][:ncat]
    jobs = [(c, m) for c in cats for m in MODELS]
    print(f"{len(combos)} phrasings x {len(cats)} categories x {len(MODELS)} "
          f"models, {len(RAW)} responses cached", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run, c, m, combos): (c, m) for c, m in jobs}
        for k, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                print(f"ERROR {futs[f]}: {str(e)[:90]}", file=sys.stderr, flush=True)
            print(f"  {k+1}/{len(jobs)} done", flush=True)
    write_json_atomic(HERE / "paraphrase_results.json", rows)
    if not rows:
        print("no usable cells")
        return

    print(f"\n{len(rows)} category-model cells")
    hv = sum(r["H_votes"] for r in rows) / len(rows)
    conf = sum(r["mean_within_prompt_confidence"] for r in rows) / len(rows)
    print(f"  mean normalized entropy of the VOTE distribution   {hv:.3f}")
    print(f"  mean within-prompt confidence in its own answer    {conf:.3f}")
    unan = sum(1 for r in rows if max(r["p"]) > 0.95)
    print(f"  cells answering identically in >95% of phrasings   {unan}/{len(rows)}")
    print("  (a high second number with a high first means each phrasing is")
    print("   decisive while the population of phrasings is not)")

    d = [r["kl_luce"] - r["kl_thur"] for r in rows]
    n = len(d)
    random.seed(4)
    B = 20000
    bs = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    print(f"\n  dKL (Luce - Thurstone) {sum(d)/n:+.4f} "
          f"[{bs[int(.025*B)]:+.4f}, {bs[int(.975*B)]:+.4f}], "
          f"Thurstone ahead in {sum(x > 0 for x in d)}/{n}")


if __name__ == "__main__":
    main()
