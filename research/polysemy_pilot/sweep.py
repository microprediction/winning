"""Broad cheap sweep: every question type, every adjective, two restriction
designs, deleting each of the top items in turn.

The point is robustness to the type of question rather than depth on any one
of them. The 99 question categories of the 2024 appendix each carry their own
prompt template and adjective list, so the sweep crosses:

    question type (99) x adjective (926 pairs) x model (cheap tier)
      x design (exclusion, two-slot) x deleted item (top 8)

Item sets are open-vocabulary: the field is whatever the model actually offers
at the unqualified prompt, filtered to alphabetic non-stopword tokens. No
hand-authored inventory is involved, so category selection cannot bias the
comparison.

Runs to a call budget, resumable, append-only. Stop it at any time with
Ctrl-C or by killing the process: every response already paid for is on disk,
and re-running continues where it stopped without re-paying. Re-analysis is
free once fetched.

Usage:  python sweep.py [budget_calls]
"""
import glob
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import key
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from datastore import RawLog, write_json_atomic
from openai import OpenAI

CLIENT = OpenAI(api_key=key())
RAW = RawLog(HERE / "sweep_raw.jsonl")
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"]  # cheap tier only
TOP_K = 8          # delete each of the top-k observed items in turn
WORKERS = 12
DEFAULT_BUDGET = 250_000

STOP = set("the and for its this that with are was very a an of to in on it is "
           "because they most two favourite favorite my best".split())

_spent = 0
_lock = threading.Lock()


def budget_left(budget):
    with _lock:
        return budget - _spent


def top20(prompt, model, budget):
    """Logged fetch that also enforces the call budget."""
    hit = RAW.get(model, prompt)
    if hit is not None:
        return hit
    global _spent
    with _lock:
        if _spent >= budget:
            raise RuntimeError("budget exhausted")
        _spent += 1
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0, messages=[{"role": "user", "content": prompt}])
    out = {t.token: math.exp(t.logprob)
           for t in r.choices[0].logprobs.content[0].top_logprobs}
    RAW.record(model, prompt, out)
    return out


def items_of(raw):
    out = {}
    for tok, p in raw.items():
        w = tok.strip().lower()
        if len(w) >= 3 and w.isalpha() and w not in STOP:
            out[w] = out.get(w, 0.0) + p
    return out


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def load_categories():
    cats = {}
    for f in sorted(glob.glob(str(HERE / "qvols" / "*.yaml"))):
        cats.update(yaml.safe_load(open(f)).get("examples", {}))
    return {k: v for k, v in cats.items()
            if v.get("adjectives") and "prompt_pair_template" in v}


def prompts(spec, adj, deleted):
    """The unqualified prompt, plus one restricted prompt per design."""
    t_orig, t_qual = spec["prompt_pair_template"][:2]
    s_orig = t_orig.replace("[MASK]", "___").replace("SOMETHING", adj)
    unq = (f'Fill in the blank with a single word: "{s_orig}" '
           "Give only the missing word.")
    excl = (f'Fill in the blank with a single word, but do not use the word '
            f'"{deleted}": "{s_orig}" Give only the missing word.')
    s_two = (t_qual.replace("[ANSWER]", deleted).replace("[MASK]", "___")
             .replace("SOMETHING", adj))
    two = (f'Fill in the blank with a single word: "{s_two}" '
           "Give only the missing word.")
    return unq, {"exclusion": excl, "two-slot": two}


def run_cell(cat, spec, adj, model, budget):
    """One (question type, adjective, model) unit: the unqualified field plus
    every deletion under both designs. Returns scored rows."""
    unq_prompt, _ = prompts(spec, adj, "x")
    u_raw = top20(unq_prompt, model, budget)
    d_u = items_of(u_raw)
    if len(d_u) < 3:
        return []
    items = sorted(d_u, key=d_u.get, reverse=True)[:10]
    uz = sum(d_u[s] for s in items)
    p_full = [d_u[s] / uz for s in items]
    a, err = calibrate_np(p_full)
    if err > 0.05:
        return []
    H = entropy_norm(p_full)

    rows = []
    for deleted in items[:TOP_K]:
        _, designs = prompts(spec, adj, deleted)
        for design, prompt in designs.items():
            try:
                q_raw = top20(prompt, model, budget)
            except RuntimeError:
                return rows  # budget hit; keep what is scored
            d_q = items_of(q_raw)
            keep = [s for s in items if s != deleted and s in d_q]
            if len(keep) < 2:
                continue
            az = sum(d_q[s] for s in keep)
            actual = [d_q[s] / az for s in keep]
            lz = sum(d_u[s] for s in keep)
            luce = [d_u[s] / lz for s in keep]
            idx = [items.index(s) for s in keep]
            w = win_probs_np(a[idx])
            thur = (w / w.sum()).tolist()
            rows.append({"category": cat, "adjective": adj, "model": model,
                         "design": design, "deleted": deleted,
                         "deleted_p": d_u[deleted] / uz, "keep": keep,
                         "actual": actual, "luce": luce, "thurstone": thur,
                         "kl_luce": kl(actual, luce), "kl_thur": kl(actual, thur),
                         "H_unq": H})
    return rows


def rep(name, ss):
    if not ss:
        print(f"{name:<40} (empty)")
        return
    import random
    diffs = [r["kl_luce"] - r["kl_thur"] for r in ss]
    n = len(diffs)
    random.seed(4)
    B = 8000
    bs = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) / n
                for _ in range(B))
    print(f"{name:<40} n={n:>6} wins={sum(d > 0 for d in diffs):>6} "
          f"mean dKL={sum(diffs)/n:+.4f} [{bs[int(.025*B)]:+.4f},{bs[int(.975*B)]:+.4f}]")


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUDGET
    cats = load_categories()
    units = [(c, s, adj, m) for c, s in cats.items()
             for adj in s["adjectives"] for m in MODELS]
    calls_per_unit = 1 + 2 * TOP_K
    print(f"{len(cats)} question types, {len(units)} units, "
          f"up to {len(units) * calls_per_unit:,} calls "
          f"({len(RAW):,} already logged), budget {budget:,}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_cell, c, s, adj, m, budget)
                for c, s, adj, m in units]
        for k, fut in enumerate(as_completed(futs)):
            try:
                rows.extend(fut.result())
            except Exception as e:
                if "budget" not in str(e):
                    print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 100 == 0:
                write_json_atomic(HERE / "sweep_results.json", rows)
                print(f"{k+1}/{len(units)} units, {len(rows):,} cells, "
                      f"{_spent:,} calls spent", flush=True)

    write_json_atomic(HERE / "sweep_results.json", rows)
    print(f"\n{len(rows):,} cells from {_spent:,} paid calls\n")
    rep("ALL", rows)
    for design in ("exclusion", "two-slot"):
        rep(f"  design={design}", [r for r in rows if r["design"] == design])
    for m in MODELS:
        rep(f"  {m}", [r for r in rows if r["model"] == m])
    rep("  non-degenerate (H>0.2)", [r for r in rows if r["H_unq"] > 0.2])
    rep("  degenerate (H<=0.2)", [r for r in rows if r["H_unq"] <= 0.2])
    rep("  favorite deleted (p>0.4)", [r for r in rows if r["deleted_p"] > 0.4])
    rep("  minor item deleted", [r for r in rows if r["deleted_p"] <= 0.4])

    # robustness across question types: how many categories favor which family
    bycat = {}
    for r in rows:
        bycat.setdefault(r["category"], []).append(r["kl_luce"] - r["kl_thur"])
    won = sum(1 for v in bycat.values() if sum(v) / len(v) > 0)
    print(f"\nquestion types favoring Thurstone: {won}/{len(bycat)}")


if __name__ == "__main__":
    main()
