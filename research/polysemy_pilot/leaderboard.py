"""Fetch the Open LLM Leaderboard and select a locally runnable model panel.

Supplies the ability axis for the scatter plot. The Thurstonian deficit is
computed from a model's own log probabilities with no labels and nothing to
train against, so if it tracks published ability scores across a wide panel it
is an unsupervised probe of capability. That claim needs an ability measure
this project did not invent, joined by exact repository name.

The leaderboard's `Type` field and `Base Model` links also identify
base/instruct pairs across the whole hub, which turns the base-versus-instruct
prediction from a hand-listed comparison into a population.

Selection favours what this machine can actually run and what the join can
trust: available on the hub, unflagged, not a merge, an architecture mlx-lm
loads, and a parameter count within a size ceiling. Results are cached to
leaderboard.json so the panel is reproducible without refetching.

Usage:  python leaderboard.py [max_params_b] [panel_size]
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from datastore import write_json_atomic

DATASET = "open-llm-leaderboard/contents"
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset={ds}"
            "&config=default&split=train&offset={off}&length={n}")
CACHE = HERE / "leaderboard.json"

# Architectures mlx-lm loads directly from original Hugging Face weights.
OK_ARCH = ("llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2", "gemma3",
           "phi", "phi3", "stablelm", "olmo", "starcoder2", "cohere",
           "internlm2", "minicpm", "granite")

ABILITY = "Average ⬆️"


def fetch_page(off, n, tries=6):
    """One page, with backoff. The endpoint rate limits, so retry rather than
    abandoning a partly fetched table."""
    url = ROWS_URL.format(ds=urllib.parse.quote(DATASET, safe=""), off=off, n=n)
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"    retry {attempt+1} at offset {off}: {str(e)[:60]}",
                  flush=True)
            time.sleep(delay)
            delay *= 2
    return None


def fetch_all(pause=0.6):
    """Resume from whatever is already cached; checkpoint every page so an
    interruption costs nothing already fetched."""
    rows = json.loads(CACHE.read_text()) if CACHE.exists() else []
    off = len(rows)
    while True:
        payload = fetch_page(off, 100)
        batch = [x["row"] for x in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        write_json_atomic(CACHE, rows)          # checkpoint before continuing
        total = payload.get("num_rows_total")
        print(f"  fetched {len(rows)}" + (f"/{total}" if total else ""), flush=True)
        off += len(batch)
        if total and len(rows) >= total:
            break
        time.sleep(pause)
    return rows


def load(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    print("fetching leaderboard ...", flush=True)
    rows = fetch_all()
    write_json_atomic(CACHE, rows)
    return rows


def runnable(r, max_params):
    arch = (r.get("Architecture") or "").lower()
    return (r.get("Available on the hub")
            and not r.get("Flagged")
            and not r.get("Merged")
            and r.get(ABILITY) is not None
            and (r.get("#Params (B)") or 1e9) <= max_params
            and any(a in arch for a in OK_ARCH))


def select(rows, max_params=9.0, panel=40):
    """Spread the panel across the ability range so the scatter has support at
    both ends rather than clustering wherever the hub is densest."""
    cands = [r for r in rows if runnable(r, max_params)]
    cands.sort(key=lambda r: r[ABILITY])
    if not cands:
        return []
    step = max(1, len(cands) // panel)
    picked = cands[::step][:panel]
    return picked


def pairs(rows, max_params=9.0):
    """Base/instruct pairs: a fine-tune whose declared base model is itself on
    the leaderboard and also runnable."""
    by_name = {r["fullname"]: r for r in rows}
    out = []
    for r in rows:
        base = (r.get("Base Model") or "").strip()
        if not base or base not in by_name:
            continue
        b = by_name[base]
        if runnable(r, max_params) and runnable(b, max_params):
            if (r.get("Type") or "").lower() != (b.get("Type") or "").lower():
                out.append((b, r))
    return out


def main():
    max_params = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
    panel_size = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rows = load()
    print(f"\n{len(rows)} leaderboard rows")
    cands = [r for r in rows if runnable(r, max_params)]
    print(f"{len(cands)} runnable here (<= {max_params}B, mlx-loadable arch, "
          f"unflagged, unmerged, scored)")

    panel = select(rows, max_params, panel_size)
    if panel:
        lo, hi = panel[0][ABILITY], panel[-1][ABILITY]
        print(f"\npanel of {len(panel)}, ability {lo:.1f} to {hi:.1f}:")
        for r in panel:
            print(f"  {r[ABILITY]:>5.1f}  {r.get('#Params (B)') or 0:>5.1f}B  "
                  f"{(r.get('Type') or '?')[:12]:<13} {r['fullname'][:58]}")
        write_json_atomic(HERE / "panel.json", panel)

    ps = pairs(rows, max_params)
    print(f"\n{len(ps)} base/instruct pairs where both sides are runnable and scored")
    for b, i in ps[:12]:
        print(f"  {b['fullname'][:44]:<45} -> {i['fullname'][:44]:<45} "
              f"{b[ABILITY]:>5.1f} -> {i[ABILITY]:>5.1f}")
    if ps:
        write_json_atomic(HERE / "panel_pairs.json",
                          [{"base": b["fullname"], "tuned": i["fullname"],
                            "base_ability": b[ABILITY], "tuned_ability": i[ABILITY],
                            "params": b.get("#Params (B)")} for b, i in ps])


if __name__ == "__main__":
    main()
