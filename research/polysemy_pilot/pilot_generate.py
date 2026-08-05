"""Pilot replication of Cekinmez et al. (arXiv 2608.00410) sense-distribution
measurement, using the claude CLI as the inference engine.

Phase 1: generation. For each word, draw N independent sentence samples.
Outputs one JSON file per word under gen/.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
GEN_DIR = HERE / "gen"
GEN_DIR.mkdir(exist_ok=True)

N_SAMPLES = 150
GEN_MODEL = "haiku"
CONCURRENCY = 16

SYSTEM = "You are a helpful assistant."


def claude_call(prompt: str, model: str) -> str:
    out = subprocess.run(
        ["claude", "-p", prompt, "--model", model,
         "--system-prompt", SYSTEM, "--tools", ""],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return out.stdout.strip()


def gen_one(word: str, i: int) -> tuple[str, int, str]:
    prompt = (f"Use the following word in a single sentence: {word}. "
              "Reply with only the sentence and nothing else.")
    return word, i, claude_call(prompt, GEN_MODEL)


def main():
    words = json.loads((HERE / "stimuli.json").read_text())
    jobs = []
    for word in words:
        out_path = GEN_DIR / f"{word}.json"
        existing = json.loads(out_path.read_text()) if out_path.exists() else []
        for i in range(len(existing), N_SAMPLES):
            jobs.append((word, i))

    print(f"{len(jobs)} generation calls to make", flush=True)
    results: dict[str, dict[int, str]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(gen_one, w, i) for w, i in jobs]
        for fut in as_completed(futs):
            try:
                word, i, sentence = fut.result()
                results.setdefault(word, {})[i] = sentence
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr, flush=True)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    for word, samples in results.items():
        out_path = GEN_DIR / f"{word}.json"
        existing = json.loads(out_path.read_text()) if out_path.exists() else []
        merged = existing + [samples[k] for k in sorted(samples)]
        out_path.write_text(json.dumps(merged, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
