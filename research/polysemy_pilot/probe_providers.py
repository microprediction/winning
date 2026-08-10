"""Probe each configured provider with one real call and report what it returns.

Resolves the capability question empirically instead of trusting vendor
documentation. For every provider with a key present in winning/.env, sends a
single one-token request asking for 20 top alternatives and classifies the
response:

  topk k=N   N alternatives came back at the position; usable directly
  chosen     only the sampled token's log probability; usable via
             continuation scoring (see providers.SCORING_NOTE)
  none       no log probabilities in the response
  ERROR ...  rejected, with the provider's message

Results are appended to provider_probe.jsonl so the panel's coverage is a
recorded fact rather than a recollection. Costs one call per model probed.

Usage:  python probe_providers.py [provider ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from datastore import append_jsonl
from providers import PROVIDERS, NO_LOGPROBS, LOCAL
from openai import OpenAI

ENV = Path("/Users/petercotton/github/winning/.env")
PROMPT = 'Fill in the blank with a single word: "My favourite bird is the ___." Give only the missing word.'


def env_keys():
    out = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def classify(resp):
    """What did this response actually give us?"""
    try:
        lp = resp.choices[0].logprobs
    except Exception:
        return "none", None
    if lp is None:
        return "none", None
    content = getattr(lp, "content", None)
    if not content:
        # completions-style: token_logprobs / top_logprobs lists
        top = getattr(lp, "top_logprobs", None)
        if top:
            return "topk", len(top[0]) if top[0] else 0
        if getattr(lp, "token_logprobs", None):
            return "chosen", 1
        return "none", None
    first = content[0]
    top = getattr(first, "top_logprobs", None)
    if top:
        return "topk", len(top)
    if getattr(first, "logprob", None) is not None:
        return "chosen", 1
    return "none", None


def probe(name, spec, keys):
    key = keys.get(spec["env_var"])
    if not key:
        return [(name, m, "skipped (no key)", None, "") for m in spec["models"][:1]]
    rows = []
    for model in spec["models"]:
        client = (OpenAI(api_key=key) if spec["base_url"] is None
                  else OpenAI(api_key=key, base_url=spec["base_url"]))
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=1, logprobs=True, top_logprobs=20,
                temperature=1.0, messages=[{"role": "user", "content": PROMPT}])
            level, k = classify(r)
            top = ""
            try:
                c = r.choices[0].logprobs.content[0]
                top = (c.top_logprobs[0].token if getattr(c, "top_logprobs", None)
                       else c.token)
            except Exception:
                pass
            rows.append((name, model, level, k, repr(top)))
        except Exception as e:
            msg = str(e).replace("\n", " ")[:150]
            rows.append((name, model, f"ERROR {msg}", None, ""))
    return rows


def main():
    keys = env_keys()
    wanted = sys.argv[1:] or list(PROVIDERS)
    print(f"{'provider':<13}{'model':<52}{'result':<26}{'k':>4}  top token")
    print("-" * 108)
    all_rows = []
    for name in wanted:
        spec = PROVIDERS.get(name)
        if not spec:
            print(f"{name:<13}unknown provider")
            continue
        for row in probe(name, spec, keys):
            n, model, level, k, top = row
            print(f"{n:<13}{model[:51]:<52}{level[:25]:<26}{k if k else '':>4}  {top}")
            all_rows.append({"provider": n, "model": model, "level": level,
                             "k": k, "top": top})
    for r in all_rows:
        append_jsonl(HERE / "provider_probe.jsonl", {"key": f"{r['provider']}||{r['model']}",
                                                     "raw": r})

    have = {r["provider"] for r in all_rows if r["level"].startswith("topk")}
    print()
    print(f"usable directly (topk): {sorted(have) or 'none yet'}")
    print(f"cannot participate:     {sorted(NO_LOGPROBS)}")
    print(f"local, no key, exact:   {sorted(LOCAL)}")


if __name__ == "__main__":
    main()
