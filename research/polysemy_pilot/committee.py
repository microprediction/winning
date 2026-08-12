"""Committees of model voters: does individual choice aggregate to a collective
Thurstone, and does withdrawal behave as Luce voters would require?

The aggregation argument runs: each voter i chooses by Luce with probabilities
p_i, votes are added, N = sum_i e_{C_i}, and by the central limit theorem

    N  ~=  m + Z,     m = sum_i p_i,   Cov(N) = sum_i [diag(p_i) - p_i p_i'],

so the plurality winner argmax_j N_j is the argmax of a correlated Gaussian: a
collective Thurstone model built from Lucean individuals.

Two things are worth separating, because only one of them is informative.

The Gaussian limit itself needs no assumption about the individual law. Any
independent voters with any choice distributions satisfy the same CLT, so
"collective Thurstone" follows from addition, not from Luce. Verifying it is a
sanity check, not a finding.

Withdrawal is where the individual law bites. Removing a candidate requires
recomputing every voter's distribution over the reduced field and re-aggregating;
one may not delete a coordinate from the aggregate Gaussian. If voters were
Lucean, the recomputation would be renormalization of each p_i, which is
predictable without asking anyone. Since Section 4 shows model voters are not
Lucean, that shortcut should fail, and the size of its failure is measurable by
re-asking the committee.

The design therefore measures, for each candidate field:

  p_i        each voter's exact distribution, read from option-token log
             probabilities with option order randomized per voter
  observed   the post-withdrawal distribution, by re-asking every voter
  luce_pred  each voter's p_i renormalized over the survivors, re-aggregated
  gauss_pred the naive shortcut: drop the coordinate from the aggregate Gaussian

and compares the three collective winner distributions.

Usage:  python committee.py [n_voters] [n_families]
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
from exact_restrict import key
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "committee_raw.jsonl")
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
LETTERS = "ABCDE"

# Fictional candidates described only by policy emphasis, so that no real
# person or party is represented and the choice is over positions.
FAMILIES = [
    {"office": "town council",
     "cands": {"Ash": "wants to expand the bus network and add cycle lanes",
               "Brill": "wants lower property taxes and a smaller budget",
               "Cove": "wants more park land and stricter building limits",
               "Dell": "wants faster housing approvals and denser building",
               "Ember": "wants more spending on schools and libraries"}},
    {"office": "sports club committee",
     "cands": {"Ash": "wants to spend on the junior programme",
               "Brill": "wants to keep fees as low as possible",
               "Cove": "wants to renovate the clubhouse",
               "Dell": "wants to recruit stronger senior players",
               "Ember": "wants more social events and fewer competitions"}},
    {"office": "library board",
     "cands": {"Ash": "wants longer opening hours",
               "Brill": "wants a larger childrens' section",
               "Cove": "wants more digital lending",
               "Dell": "wants to preserve the physical collection",
               "Ember": "wants more community meeting space"}},
]

TRAITS = [
    "You are cautious with money.", "You value fairness above efficiency.",
    "You are impatient with bureaucracy.", "You care most about children.",
    "You are a keen cyclist.", "You are retired and on a fixed income.",
    "You run a small business.", "You have lived here for forty years.",
    "You moved here last year.", "You dislike change.",
    "You are an enthusiast for new ideas.", "You care about the environment.",
]
STAKES = ["", " You have voted in every election.",
          " You rarely vote but will this time.",
          " You discussed it with neighbours first."]


def voters(n, seed=7):
    combos = list(itertools.product(range(len(TRAITS)), range(len(TRAITS)),
                                    range(len(STAKES))))
    combos = [(a, b, c) for a, b, c in combos if a < b]
    random.Random(seed).shuffle(combos)
    return combos[:n]


def persona(v):
    a, b, c = v
    return f"{TRAITS[a]} {TRAITS[b]}{STAKES[c]}"


def prompt(v, fam, names, order_seed):
    """Options are lettered and their order is randomized per voter, because
    listing order alone moves log odds by more than any effect under study."""
    order = list(names)
    random.Random(order_seed).shuffle(order)
    lines = [f"{LETTERS[i]}. {n}, who {fam['cands'][n]}"
             for i, n in enumerate(order)]
    body = "\n".join(lines)
    p = (f"{persona(v)}\n\nYou are voting for one seat on the {fam['office']}. "
         f"The candidates are:\n{body}\n\nVote for exactly one. "
         f"Answer with the letter only.\n\nVote:")
    return p, order


def _api(pr, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0, messages=[{"role": "user", "content": pr}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def ballot(v, fam, names, model, seed):
    """This voter's exact distribution over candidates."""
    pr, order = prompt(v, fam, names, seed)
    raw = RAW.fetch(model, pr, lambda: _api(pr, model))
    agg = {n: 0.0 for n in names}
    for tok, p in raw.items():
        w = tok.strip().upper()
        if len(w) == 1 and w in LETTERS[:len(order)]:
            agg[order[LETTERS.index(w)]] += p
    z = sum(agg.values())
    if z <= 0.2:
        return None
    return {n: agg[n] / z for n in names}


def winner_dist_mc(P, draws=20000, seed=1):
    """Plurality winner distribution by simulating individual votes."""
    rng = np.random.default_rng(seed)
    K = P.shape[1]
    cum = np.cumsum(P, axis=1)
    wins = np.zeros(K)
    for _ in range(draws):
        u = rng.random(P.shape[0])[:, None]
        votes = (u < cum).argmax(axis=1)
        N = np.bincount(votes, minlength=K)
        top = np.flatnonzero(N == N.max())
        wins[rng.choice(top)] += 1
    return wins / draws


def winner_dist_gauss(P, draws=20000, seed=2):
    """Gaussian-argmax prediction from the CLT moments of the vote totals."""
    rng = np.random.default_rng(seed)
    m = P.sum(axis=0)
    V = np.zeros((P.shape[1], P.shape[1]))
    for p in P:
        V += np.diag(p) - np.outer(p, p)
    L = np.linalg.cholesky(V + 1e-9 * np.eye(len(m)))
    Z = rng.standard_normal((draws, len(m))) @ L.T
    N = m + Z
    idx = N.argmax(axis=1)
    return np.bincount(idx, minlength=len(m)) / draws


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def run(fam, model, vs):
    names = list(fam["cands"])
    P = []
    for k, v in enumerate(vs):
        b = ballot(v, fam, names, model, seed=1000 + k)
        if b:
            P.append([b[n] for n in names])
    if len(P) < 0.6 * len(vs):
        return None
    P = np.array(P)

    mc = winner_dist_mc(P)
    ga = winner_dist_gauss(P)

    # withdraw the candidate the committee most favours in aggregate
    drop = int(P.sum(axis=0).argmax())
    survivors = [n for i, n in enumerate(names) if i != drop]

    # observed: re-ask every voter over the reduced field
    P2 = []
    for k, v in enumerate(vs[:len(P)]):
        b = ballot(v, fam, survivors, model, seed=2000 + k)
        if b:
            P2.append([b[n] for n in survivors])
    if len(P2) < 0.6 * len(P):
        return None
    P2 = np.array(P2)
    obs = winner_dist_mc(P2, seed=3)

    # Luce prediction: renormalize each voter, re-aggregate
    keep = [i for i in range(len(names)) if i != drop]
    PL = P[:, keep]
    PL = PL / PL.sum(axis=1, keepdims=True)
    luce = winner_dist_mc(PL, seed=4)

    # naive shortcut: drop the coordinate from the aggregate Gaussian
    gauss_drop = winner_dist_gauss(P, seed=5)[keep]
    gauss_drop = gauss_drop / gauss_drop.sum()

    return {"office": fam["office"], "model": model, "names": names,
            "n_voters": int(P.shape[0]), "dropped": names[drop],
            "survivors": survivors,
            "mean_voter_max_prob": float(P.max(axis=1).mean()),
            "mc_winner": mc.tolist(), "gauss_winner": ga.tolist(),
            "tv_clt": tv(mc, ga),
            "observed_after": obs.tolist(),
            "luce_after": luce.tolist(),
            "gauss_shortcut_after": gauss_drop.tolist(),
            "tv_luce": tv(obs, luce),
            "tv_shortcut": tv(obs, gauss_drop)}


def main():
    nv = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else len(FAMILIES)
    vs = voters(nv)
    jobs = [(f, m) for f in FAMILIES[:nf] for m in MODELS]
    print(f"{len(vs)} voters x {nf} families x {len(MODELS)} models, "
          f"{len(RAW)} cached", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run, f, m, vs) for f, m in jobs]
        for k, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                print(f"ERROR {str(e)[:100]}", file=sys.stderr, flush=True)
            print(f"  {k+1}/{len(jobs)}", flush=True)
    write_json_atomic(HERE / "committee_results.json", rows)
    if not rows:
        print("no usable committees")
        return
    print(f"\n{len(rows)} committees")
    print(f"  mean voter decisiveness (max prob)      "
          f"{sum(r['mean_voter_max_prob'] for r in rows)/len(rows):.3f}")
    print(f"  TV(individual-vote MC, Gaussian argmax) "
          f"{sum(r['tv_clt'] for r in rows)/len(rows):.4f}   <- CLT check")
    print()
    print("  after withdrawing the leading candidate, distance from observed:")
    print(f"    Luce voters renormalized and re-aggregated "
          f"{sum(r['tv_luce'] for r in rows)/len(rows):.4f}")
    print(f"    naive drop-a-coordinate shortcut          "
          f"{sum(r['tv_shortcut'] for r in rows)/len(rows):.4f}")


if __name__ == "__main__":
    main()
