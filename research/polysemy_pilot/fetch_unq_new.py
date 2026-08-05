"""Fetch unqualified 'random X' distributions for categories not yet in
random_raw.json and merge them in."""
import json, math, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_restrict import MODELS, key
from exact_analyze import INVENTORY
from openai import OpenAI
CLIENT = OpenAI(api_key=key())
PHRASES = ["Name", "Pick"]

def top20(prompt, model):
    r = CLIENT.chat.completions.create(
        model=model, max_tokens=1, logprobs=True, top_logprobs=20, temperature=1.0,
        messages=[{"role": "system", "content": "Answer with a single word and nothing else."},
                  {"role": "user", "content": prompt}])
    return {t.token: math.exp(t.logprob) for t in r.choices[0].logprobs.content[0].top_logprobs}

raw = json.loads((HERE / "random_raw.json").read_text())
jobs = [(n, ph, m) for n in INVENTORY for ph in PHRASES for m in MODELS
        if f"{n}||{ph}||{m}" not in raw]
print(f"{len(jobs)} new unqualified calls")
with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(top20, f"{ph} a random {n}.", m): (n, ph, m) for n, ph, m in jobs}
    for fut in as_completed(futs):
        n, ph, m = futs[fut]
        try:
            raw[f"{n}||{ph}||{m}"] = fut.result()
        except Exception as e:
            print(f"ERROR {n} {ph} {m}: {e}", file=sys.stderr)
(HERE / "random_raw.json").write_text(json.dumps(raw, indent=1))
print("merged")
