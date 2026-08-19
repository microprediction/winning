"""Phase 2: judge each generated sentence's sense; elicit stated distributions.

Judge follows Cekinmez et al. Appendix B prompt shape (closed label set plus
multiple/unclear/other). One judge call per word carrying all sentences.
Stated elicitation uses their 'predicting itself' sentence framing.
Outputs judged.json and stated.json.
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
JUDGE_MODEL = "sonnet"
STATED_MODEL = "haiku"  # must match the generation model: it predicts ITSELF
SYSTEM = "You are a helpful assistant."


def claude_call(prompt: str, model: str) -> str:
    out = subprocess.run(
        ["claude", "-p", prompt, "--model", model,
         "--system-prompt", SYSTEM, "--tools", ""],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return out.stdout.strip()


def extract_json(text: str):
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    return json.loads(m.group(0)) if m else None


CHUNK = 30


def judge_chunk(word: str, senses: list[str], sentences: list[str]):
    labels = senses + ["multiple", "unclear", "other"]
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        f'A language model was given only the single word "{word}" as its '
        "prompt, with no other context, and asked to use it in one sentence. "
        f"Below are {len(sentences)} sentences it produced. For each sentence, "
        f'decide which meaning of "{word}" it uses. Choose exactly one of '
        f"these labels per sentence: {labels}. "
        '"multiple" means two or more senses at once. Reply with only a JSON '
        f"array of {len(sentences)} label strings, in order, nothing else.\n\n"
        f"{numbered}"
    )
    out = extract_json(claude_call(prompt, JUDGE_MODEL))
    assert isinstance(out, list) and len(out) == len(sentences), f"bad judge output for {word}"
    return out


def judge_word(word: str, senses: list[str], sentences: list[str]):
    out = []
    for k in range(0, len(sentences), CHUNK):
        out += judge_chunk(word, senses, sentences[k:k + CHUNK])
    return word, out


def stated_word(word: str, senses: list[str]):
    prompt = (
        f'Consider the word "{word}". Its possible meanings are: {senses}. '
        "If you were asked to use this word in a single sentence 100 times, "
        "what percentage of your sentences would use each meaning? Reply with "
        "only a JSON object mapping each meaning to a percentage; they should "
        "sum to 100. Nothing else."
    )
    out = extract_json(claude_call(prompt, STATED_MODEL))
    assert isinstance(out, dict), f"bad stated output for {word}"
    return word, out


def main():
    words = json.loads((HERE / "stimuli.json").read_text())
    gen = {w: json.loads((HERE / "gen" / f"{w}.json").read_text()) for w in words}

    judged, stated = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_word, w, senses, gen[w]): ("judge", w)
                for w, senses in words.items()}
        futs |= {ex.submit(stated_word, w, senses): ("stated", w)
                 for w, senses in words.items()}
        for fut in as_completed(futs):
            kind, w = futs[fut]
            try:
                word, out = fut.result()
                (judged if kind == "judge" else stated)[word] = out
                print(f"{kind}:{word} ok", flush=True)
            except Exception as e:
                print(f"{kind}:{w} ERROR {e}", file=sys.stderr, flush=True)

    (HERE / "judged.json").write_text(json.dumps(judged, indent=1))
    (HERE / "stated.json").write_text(json.dumps(stated, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
