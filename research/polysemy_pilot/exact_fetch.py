"""Fetch exact next-token distributions for the restriction battery.
API only — analysis happens in exact_analyze.py. Saves exact_raw.json."""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import CELLS, MODELS, PHRASINGS, SYSTEM, key

CLIENT = OpenAI(api_key=key())


def top20(prompt: str, model: str) -> dict[str, float]:
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20,
        temperature=1.0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob)
            for t in r.choices[0].logprobs.content[0].top_logprobs}


def main():
    prompts = set()
    for unq, q, _ in CELLS:
        for ph in PHRASINGS:
            prompts.add(ph.format(c=unq))
            prompts.add(ph.format(c=q))
    jobs = [(p, m) for p in sorted(prompts) for m in MODELS]
    print(f"{len(jobs)} logprob calls", flush=True)

    raw = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(top20, p, m): (p, m) for p, m in jobs}
        for fut in as_completed(futs):
            p, m = futs[fut]
            try:
                raw[f"{m}||{p}"] = fut.result()
            except Exception as e:
                print(f"ERROR {m} {p}: {e}", file=sys.stderr, flush=True)
    (HERE / "exact_raw.json").write_text(json.dumps(raw, indent=1))
    print(f"saved {len(raw)} distributions", flush=True)


if __name__ == "__main__":
    main()
