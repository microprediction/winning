"""Elections: persona-conditional voting with candidate withdrawal.

Fictional candidates (two of them ideological twins), fictional voter
personas. Unrestricted vote distribution via letter... no — last-name token
logprobs, averaged over ballot orders. Withdrawal ("X has dropped out") is
the natural deletion. Tests:
  1. Luce renormalization vs Thurstone removal, KL-scored, per persona cell.
  2. Substitution: when a twin withdraws, does the other twin absorb the
     votes (correlated utilities) or do they spread proportionally (Luce)?
"""
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, key
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
N_ORDERS = 4

SCENARIOS = {
    "mayor": {
        "office": "mayor of a mid-sized city",
        "twins": ("Baker", "Klein"),
        "candidates": {
            "Baker": "cut property taxes, repair roads and bridges",
            "Klein": "lower taxes, invest in road and bridge repair",
            "Chen": "expand green public transit, build affordable housing",
            "Ortiz": "hire more police, support downtown businesses",
            "Woods": "increase school funding, expand public health clinics",
        },
    },
    "county": {
        "office": "county executive",
        "twins": ("Rivera", "Novak"),
        "candidates": {
            "Rivera": "protect parks and rivers, fund renewable energy",
            "Novak": "conserve green space, expand solar and wind programs",
            "Grant": "attract tech employers, cut business regulation",
            "Shaw": "improve rural roads, support family farms",
            "Patel": "expand county hospitals, lower prescription costs",
        },
    },
}

PERSONAS = [
    "a parent of three who wants more school funding but also owns a rental property and opposes tax rises",
    "a small business owner who hates red tape but whose child needs the county hospital's services",
    "a retired police officer who gardens, splits between public safety and park conservation",
    "a young teacher who bikes to work, torn between school funding and green transit",
    "a farmer whose fields flood, wanting both rural road repair and river protection",
    "a nurse who commutes an hour, torn between healthcare access and road repair",
]


def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def vote_dist(scen, names, persona, model, rng, dropped=None):
    cand = SCENARIOS[scen]["candidates"]
    office = SCENARIOS[scen]["office"]
    acc = {n: 0.0 for n in names}
    used = 0
    for _ in range(N_ORDERS):
        order = list(names)
        rng.shuffle(order)
        lines = [f"- {n}: {cand[n]}" for n in order]
        drop = (f" Candidate {dropped} has dropped out of the race."
                if dropped else "")
        prompt = (f"The candidates for {office}:\n" + "\n".join(lines) + "\n"
                  f"A genuinely undecided voter, {persona}, votes for one of them.{drop} "
                  "Which candidate do they vote for? "
                  "Answer with the candidate's last name only.")
        raw = top20(prompt, model)
        slot = {n: 0.0 for n in names}
        for tok, p in raw.items():
            w = tok.strip().lower()
            for n in names:
                if w == n.lower():
                    slot[n] += p
        z = sum(slot.values())
        if z > 0.2:
            for n in names:
                acc[n] += slot[n] / z
            used += 1
    if used == 0:
        return None
    return {n: acc[n] / used for n in names}


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-9)) for ai, pi in zip(a, p) if ai > 0)


def run_block(scen, persona, model, seed):
    rng = random.Random(seed)
    names = list(SCENARIOS[scen]["candidates"])
    full = vote_dist(scen, names, persona, model, rng)
    if full is None:
        return None
    p_full = [full[n] for n in names]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return None

    twins = SCENARIOS[scen]["twins"]
    cells = []
    for out in names:
        keep = [n for n in names if n != out]
        q = vote_dist(scen, keep, persona, model, rng, dropped=out)
        if q is None:
            continue
        actual = [q[n] for n in keep]
        lz = sum(full[n] for n in keep)
        luce = [full[n] / lz for n in keep]
        idx = [names.index(n) for n in keep]
        w = win_probs_np(a[idx])
        thur = (w / w.sum()).tolist()

        cell = {"scenario": scen, "persona": persona[:40], "model": model,
                "dropped": out, "dropped_p": full[out], "keep": keep,
                "actual": actual, "luce": luce, "thurstone": thur,
                "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
                "H_full": entropy_norm(p_full)}
        if out in twins:
            twin = twins[0] if out == twins[1] else twins[1]
            ti = keep.index(twin)
            cell["twin"] = twin
            cell["twin_gain_actual"] = actual[ti] - full[twin]
            cell["twin_gain_luce"] = luce[ti] - full[twin]
            cell["twin_absorb_share"] = ((actual[ti] - full[twin]) / full[out]
                                         if full[out] > 1e-6 else None)
        cells.append(cell)
    return cells


def main():
    jobs = [(s, p, m, hash((s, p, m)) & 0xFFFF)
            for s in SCENARIOS for p in PERSONAS for m in MODELS]
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_block, *j) for j in jobs]
        for fut in as_completed(futs):
            try:
                cells = fut.result()
                if cells:
                    results += cells
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    (HERE / "elections_results.json").write_text(json.dumps(results, indent=1))
    n = len(results)
    print(f"{n} withdrawal cells")

    def rep(name, ss):
        if not ss:
            return
        nn = len(ss)
        diffs = [r["kl_luce"] - r["kl_thur"] for r in ss]
        mean = sum(diffs) / nn
        random.seed(6)
        B = 20000
        boots = sorted(sum(diffs[random.randrange(nn)] for _ in range(nn)) / nn
                       for _ in range(B))
        tw = sum(d > 0 for d in diffs)
        print(f"{name:<28} n={nn:>3} Thurstone wins {tw:>3}/{nn}  "
              f"mean dKL={mean:+.4f} [{boots[int(.025*B)]:+.4f},{boots[int(.975*B)]:+.4f}]")

    rep("ALL withdrawal cells", results)
    rep("  non-degenerate (H>0.2)", [r for r in results if r["H_full"] > 0.2])
    rep("  degenerate", [r for r in results if r["H_full"] <= 0.2])

    tw_cells = [r for r in results if "twin" in r and r["twin_absorb_share"] is not None
                and r["dropped_p"] > 0.05]
    if tw_cells:
        shares = sorted(r["twin_absorb_share"] for r in tw_cells)
        luce_shares = [(r["twin_gain_luce"] / r["dropped_p"]) for r in tw_cells]
        print(f"\ntwin withdrawal (n={len(tw_cells)}, dropped share>5%):")
        print(f"  actual share of dropped votes absorbed by twin: "
              f"median {shares[len(shares)//2]:.2f}, mean {sum(shares)/len(shares):.2f}")
        print(f"  Luce-predicted absorption: mean {sum(luce_shares)/len(luce_shares):.2f}")
        print("  (substitution predicts ~1.0, Luce predicts the twin's proportional share)")


if __name__ == "__main__":
    main()
