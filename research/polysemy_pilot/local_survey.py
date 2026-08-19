"""Local survey of open-weight models with exact, untruncated item probabilities.

Running models on this machine removes the two compromises the API batteries
live with. There is no top-20 window, because the whole vocabulary is
available, and there is no sub-word matching problem, because each candidate
item is scored by teacher forcing: the item's full token sequence is appended
to the prompt and its log probabilities are summed. An item's probability is
therefore measured rather than inferred from whichever of its tokens happened
to surface in a truncated list.

It also buys the two experiments the API panel cannot run:

  1. base against instruct. If what separates model generations is
     post-training rather than scale, a base checkpoint should obey the Choice
     Axiom more closely than its instruction-tuned sibling. Both are given the
     identical raw completion prompt, with no chat template on either side, so
     the only difference is the checkpoint.

  2. a panel wide enough to correlate. The Thurstonian deficit is computed
     from a model's own log probabilities with no labels, so if it tracks
     published ability scores across twenty models it is an unsupervised
     probe of capability. Five models cannot establish that; this can.

Prompts are the completion-shaped stimuli of the 2024 version, which suit base
models natively:

    unqualified   "My favourite bird is the"
    two-slot      "My two favourite birds are the eagle and the"

Append-only, resumable, and free. Usage:  python local_survey.py [model ...]
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_analyze import calibrate_np, win_probs_np, entropy_norm
from inventory import INVENTORY
from datastore import append_jsonl, load_jsonl, write_json_atomic

import mlx.core as mx
import numpy as np
from mlx_lm import load

RAW_LOG = HERE / "local_raw.jsonl"
OPEN_LOG = HERE / "local_open_raw.jsonl"
TOP_K = 6          # delete each of the top-k items in turn
MAX_ITEMS = 10     # candidate field size per category
STOP = set("the and for its this that with are was very a an of to in on it is "
           "because they most two favourite favorite my best one same other "
           "than not but you all any some more".split())

# (repo, family, size_b, tuning). Base and instruct siblings share a family
# and size so they pair automatically.
MODELS = [
    ("mlx-community/Qwen2.5-0.5B-4bit",            "qwen2.5", 0.5, "base"),
    ("mlx-community/Qwen2.5-0.5B-Instruct-4bit",   "qwen2.5", 0.5, "instruct"),
    ("mlx-community/Qwen2.5-1.5B-4bit",            "qwen2.5", 1.5, "base"),
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit",   "qwen2.5", 1.5, "instruct"),
    ("mlx-community/Qwen2.5-7B-4bit",              "qwen2.5", 7.0, "base"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit",     "qwen2.5", 7.0, "instruct"),
    ("mlx-community/Llama-3.2-1B-4bit",            "llama3.2", 1.0, "base"),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit",   "llama3.2", 1.0, "instruct"),
    ("mlx-community/Llama-3.2-3B-4bit",            "llama3.2", 3.0, "base"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit",   "llama3.2", 3.0, "instruct"),
    ("mlx-community/Meta-Llama-3.1-8B-4bit",       "llama3.1", 8.0, "base"),
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1", 8.0, "instruct"),
    ("mlx-community/Mistral-7B-v0.3-4bit",         "mistral", 7.0, "base"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral", 7.0, "instruct"),
    ("mlx-community/gemma-2-2b-4bit",              "gemma2", 2.0, "base"),
    ("mlx-community/gemma-2-2b-it-4bit",           "gemma2", 2.0, "instruct"),
]

PLURAL_FIX = {"letter of the alphabet": "letters of the alphabet",
              "state in the u.s.": "states in the u.s.",
              "day of the week": "days of the week",
              "unit of length": "units of length"}


def plural(cat):
    if cat in PLURAL_FIX:
        return PLURAL_FIX[cat]
    words = cat.split()
    w = words[-1]
    if w.endswith(("s", "x", "z", "ch", "sh")):
        w += "es"
    elif len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
        w = w[:-1] + "ies"
    else:
        w += "s"
    return " ".join(words[:-1] + [w])


class Scorer:
    """Exact conditional item probabilities by teacher forcing."""

    def __init__(self, repo):
        self.model, self.tok = load(repo)

    def item_logprobs(self, prompt, items):
        """log P(item | prompt) for each item, summed over the item's tokens."""
        p_ids = self.tok.encode(prompt)
        out = []
        for item in items:
            i_ids = self.tok.encode(" " + item, add_special_tokens=False)
            if not i_ids:
                out.append(-1e9)
                continue
            ids = mx.array([p_ids + i_ids])
            logits = self.model(ids)[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            total = 0.0
            for j, tid in enumerate(i_ids):
                pos = len(p_ids) + j - 1     # distribution that predicts tid
                total += float(lp[pos, tid])
            out.append(total)
        return out

    def dist(self, prompt, items):
        """Normalized exact distribution over the candidate item set."""
        lps = self.item_logprobs(prompt, items)
        m = max(lps)
        w = [math.exp(x - m) for x in lps]
        z = sum(w)
        return [x / z for x in w]

    def next_token_probs(self, prompt):
        """Full next-token distribution as a numpy vector."""
        ids = mx.array([self.tok.encode(prompt)])
        probs = mx.softmax(self.model(ids)[0, -1].astype(mx.float32))
        mx.eval(probs)
        return np.array(probs)

    def open_field(self, prompt, max_items=10):
        """Open-vocabulary item field: the words the model itself offers.

        Mirrors the API batteries, which read first tokens from a truncated
        list, except that nothing is truncated here. Without this mode the
        local panel is not comparable to those batteries: an inventory-
        restricted design compresses the divergence between the two families
        by roughly an order of magnitude, because it discards exactly the tail
        where they disagree.
        """
        pr = self.next_token_probs(prompt)
        agg = {}
        for idx in np.argsort(pr)[-400:][::-1]:
            w = self.tok.decode([int(idx)]).strip().lower()
            if len(w) >= 3 and w.isalpha() and w not in STOP:
                agg[w] = agg.get(w, 0.0) + float(pr[idx])
        items = sorted(agg, key=agg.get, reverse=True)[:max_items]
        return items, [agg[i] for i in items]

    def open_restricted(self, prompt, items):
        """Probability of each already-chosen item at a restricted prompt."""
        pr = self.next_token_probs(prompt)
        agg = {i: 0.0 for i in items}
        for idx in np.argsort(pr)[-400:][::-1]:
            w = self.tok.decode([int(idx)]).strip().lower()
            if w in agg:
                agg[w] += float(pr[idx])
        return [agg[i] for i in items]


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def run_model(repo, family, size, tuning, cache):
    key_pre = repo
    todo = [c for c in INVENTORY if f"{key_pre}||{c}" not in cache]
    if not todo:
        print(f"  {repo}: fully cached", flush=True)
        return
    print(f"  loading {repo} ...", flush=True)
    try:
        sc = Scorer(repo)
    except Exception as e:
        print(f"  SKIP {repo}: {str(e)[:120]}", flush=True)
        append_jsonl(RAW_LOG, {"key": f"{key_pre}||__error__", "raw":
                               {"error": str(e)[:300]}})
        return
    for ci, cat in enumerate(todo):
        items = INVENTORY[cat][:MAX_ITEMS]
        pl = plural(cat)
        unq = f"My favourite {cat} is the"
        p_unq = sc.dist(unq, items)
        rec = {"key": f"{key_pre}||{cat}", "repo": repo, "family": family,
               "size": size, "tuning": tuning, "category": cat,
               "items": items, "unqualified": p_unq, "restricted": {}}
        order = sorted(range(len(items)), key=lambda i: -p_unq[i])
        for oi in order[:TOP_K]:
            fav = items[oi]
            two = f"My two favourite {pl} are the {fav} and the"
            rec["restricted"][fav] = sc.dist(two, items)
        append_jsonl(RAW_LOG, rec)
        cache[rec["key"]] = rec
        if (ci + 1) % 10 == 0:
            print(f"    {ci+1}/{len(todo)} categories", flush=True)


def run_model_open(repo, family, size, tuning, cache):
    """Open-vocabulary pass: one forward pass per prompt, field taken from the
    model's own unqualified answer. Comparable to the API batteries."""
    todo = [c for c in INVENTORY if f"{repo}||{c}" not in cache]
    if not todo:
        print(f"  {repo}: fully cached (open)", flush=True)
        return
    print(f"  loading {repo} (open-vocabulary) ...", flush=True)
    try:
        sc = Scorer(repo)
    except Exception as e:
        print(f"  SKIP {repo}: {str(e)[:120]}", flush=True)
        append_jsonl(OPEN_LOG, {"key": f"{repo}||__error__",
                                "raw": {"error": str(e)[:300]}})
        return
    for ci, cat in enumerate(todo):
        pl = plural(cat)
        items, p_unq = sc.open_field(f"My favourite {cat} is the", MAX_ITEMS)
        if len(items) < 3:
            append_jsonl(OPEN_LOG, {"key": f"{repo}||{cat}", "repo": repo,
                                    "family": family, "size": size,
                                    "tuning": tuning, "category": cat,
                                    "items": items, "unqualified": p_unq,
                                    "restricted": {}})
            continue
        z = sum(p_unq)
        p_unq = [x / z for x in p_unq]
        rec = {"key": f"{repo}||{cat}", "repo": repo, "family": family,
               "size": size, "tuning": tuning, "category": cat,
               "items": items, "unqualified": p_unq, "restricted": {}}
        for oi in sorted(range(len(items)), key=lambda i: -p_unq[i])[:TOP_K]:
            fav = items[oi]
            raw = sc.open_restricted(
                f"My two favourite {pl} are the {fav} and the", items)
            zz = sum(raw)
            if zz > 0:
                rec["restricted"][fav] = [x / zz for x in raw]
        append_jsonl(OPEN_LOG, rec)
        cache[rec["key"]] = rec
        if (ci + 1) % 25 == 0:
            print(f"    {ci+1}/{len(todo)} categories", flush=True)


def score(rec):
    """Luce renormalization against Thurstonian removal, exact inputs."""
    items, p_unq = rec["items"], rec["unqualified"]
    if len(items) < 3:
        return []
    a, err = calibrate_np(p_unq)
    if err > 0.05:
        return []
    H = entropy_norm(p_unq)
    rows = []
    for fav, p_res in rec["restricted"].items():
        fi = items.index(fav)
        keep = [i for i in range(len(items)) if i != fi]
        az = sum(p_res[i] for i in keep)
        lz = sum(p_unq[i] for i in keep)
        if az <= 0 or lz <= 0:
            continue
        actual = [p_res[i] / az for i in keep]
        luce = [p_unq[i] / lz for i in keep]
        w = win_probs_np(a[keep])
        thur = (w / w.sum()).tolist()
        rows.append({"repo": rec["repo"], "family": rec["family"],
                     "size": rec["size"], "tuning": rec["tuning"],
                     "category": rec["category"], "deleted": fav,
                     "deleted_p": p_unq[fi], "kl_luce": kl(actual, luce),
                     "kl_thur": kl(actual, thur), "H_unq": H})
    return rows


def report(rows):
    import random

    def stat(ss):
        d = [r["kl_luce"] - r["kl_thur"] for r in ss]
        n = len(d)
        random.seed(4)
        B = 8000
        bs = sorted(sum(d[random.randrange(n)] for _ in range(n)) / n
                    for _ in range(B))
        return sum(d) / n, bs[int(.025 * B)], bs[int(.975 * B)], n

    print(f"\n{'model':<48}{'tuning':<10}{'n':>5}{'dKL':>9}  95% CI")
    byrepo = {}
    for r in rows:
        byrepo.setdefault(r["repo"], []).append(r)
    for repo, ss in sorted(byrepo.items()):
        m, lo, hi, n = stat(ss)
        print(f"{repo[:47]:<48}{ss[0]['tuning']:<10}{n:>5}{m:>+9.3f}  "
              f"[{lo:+.3f},{hi:+.3f}]")

    print("\nbase against instruct, paired within family, size and category")
    byfam = {}
    for r in rows:
        byfam.setdefault((r["family"], r["size"]), {}).setdefault(
            r["tuning"], {})[(r["category"], r["deleted"])] = r
    alld = []
    for (fam, size), d in sorted(byfam.items()):
        if "base" not in d or "instruct" not in d:
            continue
        shared = set(d["base"]) & set(d["instruct"])
        if not shared:
            continue
        diffs = [(d["instruct"][k]["kl_luce"] - d["instruct"][k]["kl_thur"])
                 - (d["base"][k]["kl_luce"] - d["base"][k]["kl_thur"])
                 for k in shared]
        alld += diffs
        print(f"  {fam} {size}B: n={len(diffs):>4} "
              f"instruct minus base = {sum(diffs)/len(diffs):+.4f}  "
              f"positive in {sum(x > 0 for x in diffs)}/{len(diffs)}")
    if alld:
        m, lo, hi, n = stat([{"kl_luce": x, "kl_thur": 0.0} for x in alld])
        print(f"  POOLED: n={n} instruct minus base = {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
        print("  positive means instruction tuning moves the model AWAY from Luce")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    open_mode = "--open" in sys.argv
    log = OPEN_LOG if open_mode else RAW_LOG
    out = "local_open_results.json" if open_mode else "local_results.json"
    cache = load_jsonl(log, key="key")
    todo = [m for m in MODELS if not args or any(w in m[0] for w in args)]
    print(f"{len(todo)} models, {len(INVENTORY)} categories, "
          f"{len(cache)} records cached, "
          f"mode={'open-vocabulary' if open_mode else 'inventory'}")
    for repo, family, size, tuning in todo:
        if open_mode:
            run_model_open(repo, family, size, tuning, cache)
        else:
            run_model(repo, family, size, tuning, cache)

    rows = []
    for rec in cache.values():
        if "__error__" in rec.get("key", ""):
            continue
        rows.extend(score(rec))
    write_json_atomic(HERE / out, rows)
    print(f"\n{len(rows)} scored cells")
    if rows:
        report(rows)


if __name__ == "__main__":
    main()
