"""Exact-probability choice-set restriction test on OpenAI models.

Port of the qualified-question design of Cotton (2024) to autoregressive
chat models, with EXACT next-token distributions from logprobs (no sampling
noise). For each category cell:

  unqualified prompt:  "My favorite color is"        -> top-20 logprobs
  qualified prompt:    "My favorite warm color is"   -> top-20 logprobs

Items are counted only when they appear as complete single tokens in the
unqualified top-20 (collided prefixes are reported as excluded mass).
Zero-parameter predictions of the qualified distribution over the subset:

  Luce:      renormalize unqualified probabilities over the subset
  Thurstone: calibrate unit-noise locations to the full unqualified item
             distribution, drop non-subset contestants, recompute win probs
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pilot_analyze import calibrate_locations, win_probs, rmse, entropy_norm

from openai import OpenAI

HERE = Path(__file__).parent
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
PHRASINGS = ["My favorite {c} is", "The best {c} is"]

# category cell: (unqualified noun, qualified noun, subset membership)
CELLS = [
    ("color", "warm color", ["red", "orange", "yellow", "pink"]),
    ("color", "cool color", ["blue", "green", "purple", "teal"]),
    ("fruit", "tropical fruit", ["mango", "banana", "pineapple", "papaya", "kiwi", "coconut"]),
    ("animal", "African animal", ["elephant", "lion", "giraffe", "zebra", "cheetah", "hippo", "rhino"]),
    ("animal", "domestic animal", ["dog", "cat", "horse", "rabbit", "hamster"]),
    ("musical instrument", "string instrument", ["guitar", "violin", "cello", "harp", "banjo"]),
    ("planet", "gas giant planet", ["Jupiter", "Saturn", "Uranus", "Neptune"]),
    ("metal", "precious metal", ["gold", "silver", "platinum"]),
    ("bird", "bird of prey", ["eagle", "hawk", "owl", "falcon"]),
    ("flower", "spring flower", ["tulip", "daffodil", "lily", "daisy", "crocus"]),
    ("vegetable", "root vegetable", ["carrot", "potato", "beet", "radish", "turnip"]),
    ("tree", "evergreen tree", ["pine", "cedar", "spruce", "fir"]),
    ("sport", "team sport", ["soccer", "basketball", "football", "baseball", "hockey", "volleyball"]),
    ("hot drink", "caffeinated hot drink", ["coffee", "tea"]),
    ("month", "winter month", ["December", "January", "February"]),
    ("month", "summer month", ["June", "July", "August"]),
    ("day of the week", "weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
    ("letter of the alphabet", "vowel", ["A", "E", "I", "O", "U"]),
    ("state in the U.S.", "Western state in the U.S.",
     ["California", "Colorado", "Oregon", "Washington", "Arizona", "Nevada",
      "Montana", "Idaho", "Wyoming", "Utah", "Alaska", "Hawaii", "Texas"]),
    ("gemstone", "blue gemstone", ["sapphire", "topaz", "aquamarine", "turquoise"]),
]

SYSTEM = "Complete the sentence with a single word and nothing else."


def key():
    for line in Path("/Users/petercotton/github/winning/.env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no key")


CLIENT = OpenAI(api_key=key())


def top20(prompt: str, model: str) -> dict[str, float]:
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}])
    out = {}
    for t in r.choices[0].logprobs.content[0].top_logprobs:
        out[t.token] = math.exp(t.logprob)
    return out


def match_items(dist: dict[str, float]) -> dict[str, float]:
    """Keep tokens that are complete words (full item names), case-folded."""
    matched = {}
    for tok, p in dist.items():
        w = tok.strip()
        if len(w) >= 1 and (w.isalpha() or w.isupper()):
            matched[w.lower()] = matched.get(w.lower(), 0.0) + p
    return matched


def run_cell(cell, phrasing, model):
    unq_noun, q_noun, subset = cell
    subset_l = [s.lower() for s in subset]
    p_unq = top20(phrasing.format(c=unq_noun), model)
    p_q = top20(phrasing.format(c=q_noun), model)

    m_unq = match_items(p_unq)
    m_q = match_items(p_q)

    # analysis subset: declared members observed in BOTH distributions
    common = [s for s in subset_l if s in m_unq and s in m_q]
    if len(common) < 2:
        return None

    # actual qualified distribution over subset
    az = sum(m_q[s] for s in common)
    actual = [m_q[s] / az for s in common]

    # full unqualified item set for Thurstone calibration
    items = sorted(m_unq, key=m_unq.get, reverse=True)
    uz = sum(m_unq.values())
    p_full = [m_unq[s] / uz for s in items]

    # Luce: renormalize unqualified over subset
    lz = sum(m_unq[s] for s in common)
    luce_pred = [m_unq[s] / lz for s in common]

    # Thurstone: calibrate on full set, drop non-subset contestants
    a = calibrate_locations(p_full)
    idx = [items.index(s) for s in common]
    thur_pred = win_probs([a[i] for i in idx])

    return {
        "cell": f"{unq_noun} -> {q_noun}", "phrasing": phrasing, "model": model,
        "subset": common, "actual": actual, "luce": luce_pred, "thurstone": thur_pred,
        "rmse_luce": rmse(list(zip(luce_pred, actual))),
        "rmse_thurstone": rmse(list(zip(thur_pred, actual))),
        "H_unq": entropy_norm(p_full),
        "matched_mass_unq": uz, "matched_mass_q": sum(m_q.values()),
        "n_subset": len(common),
    }


def main():
    jobs = [(c, ph, m) for c in CELLS for ph in PHRASINGS for m in MODELS]
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_cell, c, ph, m) for c, ph, m in jobs]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    results.sort(key=lambda r: (r["model"], r["cell"], r["phrasing"]))

    (HERE / "exact_results.json").write_text(json.dumps(results, indent=1))

    print(f"{len(results)} usable cells of {len(jobs)}")
    for m in MODELS:
        rs = [r for r in results if r["model"] == m]
        tw = sum(r["rmse_thurstone"] < r["rmse_luce"] for r in rs)
        pl = rmse([pq for r in rs for pq in zip(r["luce"], r["actual"])])
        pt = rmse([pq for r in rs for pq in zip(r["thurstone"], r["actual"])])
        print(f"{m:<12} cells={len(rs):>3}  Thurstone wins {tw}/{len(rs)}  "
              f"pooled RMSE: Luce={pl:.4f} Thurstone={pt:.4f}")

    # paired bootstrap over cells (pooled across models)
    import random
    random.seed(1234)
    diffs = [r["rmse_luce"] - r["rmse_thurstone"] for r in results]
    n = len(diffs)
    wins = 0
    B = 10000
    for _ in range(B):
        s = sum(diffs[random.randrange(n)] for _ in range(n))
        if s > 0:
            wins += 1
    print(f"\nmean RMSE difference (Luce - Thurstone): {sum(diffs)/n:+.4f}")
    print(f"bootstrap P(Thurstone better) = {wins/B:.4f}  (n={n} cells)")


if __name__ == "__main__":
    main()
