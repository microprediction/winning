"""Phase 4: choice-set restriction — the Luce-vs-Thurstone discriminating test.

Words are selected dynamically: any word whose (large-sample) unqualified
distribution puts less than 95% on the modal sense. The modal sense is
excluded by instruction and we compare zero-parameter predictions of the
restricted distribution:

  Luce:      renormalize the unqualified distribution over remaining senses
  Thurstone: calibrate unit-noise locations to the unqualified distribution,
             remove the modal contestant, recompute win probabilities

Both use add-half smoothed unqualified counts. Resumable: tops up each
word's restricted samples to N_SAMPLES.
"""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pilot_analyze import calibrate_locations, win_probs, rmse
from pilot_judge import judge_word as judge_word_chunked

HERE = Path(__file__).parent
OUT_DIR = HERE / "restricted"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLES = 150
GEN_MODEL = "haiku"
SYSTEM = "You are a helpful assistant."
MODAL_CAP = 0.95  # word usable if modal sense below this

GLOSS = {
    "bolt": {"fastener": "a metal fastener", "lightning": "lightning",
             "run": "running or fleeing", "lock": "locking a door"},
    "bank": {"finance": "a financial institution", "river": "a riverbank",
             "tilt": "tilting or banking a turn", "heap": "a heap or mass"},
    "spring": {"season": "the season", "coil": "a metal coil",
               "water": "a water source", "leap": "leaping or jumping"},
    "mole": {"animal": "the burrowing animal", "skin": "a skin blemish",
             "spy": "a spy or infiltrator", "chemistry": "the chemistry unit",
             "sauce": "the Mexican sauce"},
    "seal": {"animal": "the animal", "close": "closing or sealing something",
             "stamp": "a stamp or emblem"},
    "pitch": {"throw": "throwing a ball", "tone": "musical or vocal pitch",
              "sales": "a sales pitch or pitching an idea",
              "field": "a sports field"},
    "crane": {"bird": "the bird", "machine": "the lifting machine",
              "stretch": "craning or stretching one's neck"},
    "bar": {"pub": "a pub or drinking establishment", "rod": "a rod or beam",
            "block": "blocking or prohibiting", "law": "the legal profession",
            "music": "a musical measure"},
    "jaguar": {"animal": "the animal", "car": "the car brand"},
    "python": {"snake": "the snake", "language": "the programming language"},
    "bat": {"animal": "the flying animal", "sports": "sports equipment",
            "hit": "hitting or striking"},
    "pen": {"writing": "the writing instrument", "enclosure": "an animal enclosure",
            "prison": "prison or jail"},
    "match": {"firestick": "the fire-starting stick", "contest": "a contest or game",
              "pairing": "matching or pairing things"},
    "iron": {"metal": "the metal element", "appliance": "the clothes-pressing appliance or ironing clothes",
             "press": "pressing or smoothing", "golf": "the golf club"},
    "port": {"harbor": "a harbor", "left": "the left side of a ship",
             "wine": "the fortified wine", "connector": "a computer port or connector"},
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


def unqualified_dist(word, senses, judged):
    counts = {s: 0.5 for s in senses}  # add-half smoothing
    for lab in judged[word]:
        k = str(lab).strip().lower()
        if k in counts:
            counts[k] += 1
    z = sum(counts.values())
    return [counts[s] / z for s in senses]


def pick_targets(stimuli, judged):
    targets = {}
    for w, senses in stimuli.items():
        p = unqualified_dist(w, senses, judged)
        modal = senses[p.index(max(p))]
        if max(p) < MODAL_CAP:
            targets[w] = (modal, GLOSS[w][modal])
    return targets


def gen_one(word, gloss):
    prompt = (f"Use the following word in a single sentence: {word}. "
              f"Do not use it in the sense of {gloss}. "
              "Reply with only the sentence and nothing else.")
    return word, claude_call(prompt, GEN_MODEL)


def main():
    stimuli = json.loads((HERE / "stimuli.json").read_text())
    judged = json.loads((HERE / "judged.json").read_text())
    targets = pick_targets(stimuli, judged)
    (HERE / "restriction_targets.json").write_text(json.dumps(
        {w: t[0] for w, t in targets.items()}, indent=1))
    print(f"usable words ({len(targets)}): "
          + ", ".join(f"{w}(-{t[0]})" for w, t in targets.items()), flush=True)

    # --- top up restricted samples
    jobs = []
    for w, (_, gloss) in targets.items():
        f = OUT_DIR / f"{w}.json"
        have = len(json.loads(f.read_text())) if f.exists() else 0
        jobs += [(w, gloss)] * max(0, N_SAMPLES - have)
    print(f"{len(jobs)} restricted generations", flush=True)
    got = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(gen_one, w, g) for w, g in jobs]
        for k, fut in enumerate(as_completed(futs)):
            try:
                w, s = fut.result()
                got.setdefault(w, []).append(s)
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
            if (k + 1) % 50 == 0:
                print(f"{k+1}/{len(jobs)}", flush=True)
    for w, sents in got.items():
        f = OUT_DIR / f"{w}.json"
        existing = json.loads(f.read_text()) if f.exists() else []
        f.write_text(json.dumps(existing + sents, indent=1))

    # --- judge restricted samples (chunked, parallel over words)
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_word_chunked, w, stimuli[w],
                          json.loads((OUT_DIR / f"{w}.json").read_text())): w
                for w in targets}
        for fut in as_completed(futs):
            w = futs[fut]
            try:
                _, labels = fut.result()
                results[w] = labels
                print(f"judged {w}", flush=True)
            except Exception as e:
                print(f"judge {w} ERROR {e}", file=sys.stderr, flush=True)
    (HERE / "restricted_judged.json").write_text(json.dumps(results, indent=1))

    # --- compare predictions
    print(f"\n{'word':<7} {'Luce':>8} {'Thurstone':>10}   actual restricted dist")
    report = {}
    for w, (excl, _) in targets.items():
        if w not in results:
            continue
        senses = stimuli[w]
        p_uq = unqualified_dist(w, senses, judged)
        keep = [s for s in senses if s != excl]
        ki = [senses.index(s) for s in keep]

        rc = {s: 0 for s in keep}
        for lab in results[w]:
            k = str(lab).strip().lower()
            if k in rc:
                rc[k] += 1
        rz = sum(rc.values())
        if rz == 0:
            continue
        actual = [rc[s] / rz for s in keep]

        lz = sum(p_uq[i] for i in ki)
        luce_pred = [p_uq[i] / lz for i in ki]
        a = calibrate_locations(p_uq)
        thur_pred = win_probs([a[i] for i in ki])

        le = rmse(list(zip(luce_pred, actual)))
        te = rmse(list(zip(thur_pred, actual)))
        report[w] = {"exclude": excl, "keep": keep, "n": rz, "actual": actual,
                     "luce": luce_pred, "thurstone": thur_pred,
                     "counts": rc, "rmse_luce": le, "rmse_thurstone": te}
        print(f"{w:<7} {le:>8.4f} {te:>10.4f}   "
              + ", ".join(f"{s}:{x:.2f}" for s, x in zip(keep, actual)))

    (HERE / "restriction_analysis.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
