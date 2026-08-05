"""Phase 4: choice-set restriction — the Luce-vs-Thurstone discriminating test.

For words with non-degenerate generated distributions, re-sample with the
modal sense excluded, then compare zero-parameter predictions of the
restricted distribution:

  Luce:      renormalize the unqualified distribution over remaining senses
  Thurstone: calibrate unit-noise locations to the unqualified distribution,
             remove the modal contestant, recompute win probabilities

Both use add-half smoothed unqualified counts. Mirrors the qualified-question
design of Cotton (2024) 'A Paradox in Machine Preference'.
"""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pilot_analyze import calibrate_locations, win_probs, rmse

HERE = Path(__file__).parent
OUT_DIR = HERE / "restricted"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLES = 30
GEN_MODEL = "haiku"
JUDGE_MODEL = "sonnet"
SYSTEM = "You are a helpful assistant."

# word -> (modal sense to exclude, natural-language gloss of that sense)
TARGETS = {
    "bolt": ("lightning", "lightning"),
    "seal": ("animal", "the animal"),
    "pitch": ("sales", "a sales pitch or pitching an idea"),
    "bat": ("hit", "hitting or striking"),
    "iron": ("appliance", "the clothes-pressing appliance or ironing clothes"),
}


def claude_call(prompt: str, model: str) -> str:
    out = subprocess.run(
        ["claude", "-p", prompt, "--model", model,
         "--system-prompt", SYSTEM, "--tools", ""],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return out.stdout.strip()


def gen_one(word: str, gloss: str, i: int):
    prompt = (f"Use the following word in a single sentence: {word}. "
              f"Do not use it in the sense of {gloss}. "
              "Reply with only the sentence and nothing else.")
    return word, i, claude_call(prompt, GEN_MODEL)


def judge_word(word, senses, sentences):
    import re
    labels = senses + ["multiple", "unclear", "other"]
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        f'A language model was asked to use the word "{word}" in one sentence. '
        f"Below are {len(sentences)} sentences it produced. For each sentence, "
        f'decide which meaning of "{word}" it uses. Choose exactly one of '
        f"these labels per sentence: {labels}. "
        f"Reply with only a JSON array of {len(sentences)} label strings, in order.\n\n"
        f"{numbered}"
    )
    txt = claude_call(prompt, JUDGE_MODEL)
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    out = json.loads(m.group(0))
    assert len(out) == len(sentences)
    return out


def main():
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    judged = json.loads((HERE / "judged.json").read_text())

    # --- generate restricted samples
    jobs = [(w, gloss, i) for w, (_, gloss) in TARGETS.items()
            for i in range(N_SAMPLES)
            if not (OUT_DIR / f"{w}.json").exists()]
    print(f"{len(jobs)} restricted generations", flush=True)
    got = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(gen_one, w, g, i) for w, g, i in jobs]
        for k, fut in enumerate(as_completed(futs)):
            try:
                w, i, s = fut.result()
                got.setdefault(w, []).append(s)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 25 == 0:
                print(f"{k+1}/{len(jobs)}", flush=True)
    for w, sents in got.items():
        (OUT_DIR / f"{w}.json").write_text(json.dumps(sents, indent=1))

    # --- judge restricted samples
    results = {}
    for w in TARGETS:
        senses = stimuli[w]
        sents = json.loads((OUT_DIR / f"{w}.json").read_text())
        labels = judge_word(w, senses, sents)
        results[w] = labels
        print(f"judged {w}", flush=True)
    (HERE / "restricted_judged.json").write_text(json.dumps(results, indent=1))

    # --- compare predictions
    print(f"\n{'word':<7} {'Luce':>8} {'Thurstone':>10}   actual restricted dist")
    luce_all, thur_all = [], []
    report = {}
    for w, (excl, _) in TARGETS.items():
        senses = stimuli[w]
        # unqualified counts, add-half smoothing
        uq = {s: 0.5 for s in senses}
        for lab in judged[w]:
            k = str(lab).strip().lower()
            if k in uq:
                uq[k] += 1
        z = sum(uq.values())
        p_uq = [uq[s] / z for s in senses]

        keep = [s for s in senses if s != excl]
        ki = [senses.index(s) for s in keep]

        # actual restricted distribution
        rc = {s: 0 for s in keep}
        for lab in results[w]:
            k = str(lab).strip().lower()
            if k in rc:
                rc[k] += 1
        rz = sum(rc.values())
        actual = [rc[s] / rz for s in keep]

        # Luce: renormalize
        lz = sum(p_uq[i] for i in ki)
        luce_pred = [p_uq[i] / lz for i in ki]

        # Thurstone: calibrate, drop contestant, recompute
        a = calibrate_locations(p_uq)
        thur_pred = win_probs([a[i] for i in ki])

        le = rmse(list(zip(luce_pred, actual)))
        te = rmse(list(zip(thur_pred, actual)))
        luce_all += list(zip(luce_pred, actual))
        thur_all += list(zip(thur_pred, actual))
        report[w] = {"exclude": excl, "keep": keep, "actual": actual,
                     "luce": luce_pred, "thurstone": thur_pred,
                     "rmse_luce": le, "rmse_thurstone": te}
        print(f"{w:<7} {le:>8.4f} {te:>10.4f}   "
              + ", ".join(f"{s}:{x:.2f}" for s, x in zip(keep, actual)))

    print(f"\npooled RMSE:  Luce={rmse(luce_all):.4f}  Thurstone={rmse(thur_all):.4f}")
    (HERE / "restriction_analysis.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
