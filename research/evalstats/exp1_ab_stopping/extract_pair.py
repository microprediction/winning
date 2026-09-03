"""Extract aligned per-item outcomes for two models' MMLU-Pro runs
from the Open LLM Leaderboard details datasets. 330 MB of prompts
per model reduced to one npz of aligned binary vectors."""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get(
    "SCRATCH", "/private/tmp/claude-501/-Users-petercotton-github-winning/"
    "4cfe1164-1ade-46c7-a29a-33bedf35fe90/scratchpad")

RUNS = {
    "mistral7b": ("open-llm-leaderboard/mistralai__Mistral-7B-v0.1-details",
                  "mistralai__Mistral-7B-v0.1/"
                  "samples_leaderboard_mmlu_pro_2024-06-16T16-57-41.377142.json"),
    "qwen15_7b": ("open-llm-leaderboard/Qwen__Qwen1.5-7B-details",
                  "Qwen__Qwen1.5-7B/"
                  "samples_leaderboard_mmlu_pro_2024-06-16T19-09-42.860141.json"),
}


def read_samples(path):
    out = {}
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = (json.loads(line) for line in f if line.strip())
        for r in records:
            doc_id = r.get("doc_id")
            acc = r.get("acc", r.get("exact_match"))
            if doc_id is None or acc is None:
                keys = sorted(r.keys())
                raise SystemExit(f"unexpected record keys: {keys}")
            out[int(doc_id)] = int(round(float(acc)))
    return out


if __name__ == "__main__":
    vecs = {}
    for name, (repo, fname) in RUNS.items():
        p = hf_hub_download(repo, fname, repo_type="dataset",
                            local_dir=os.path.join(SCRATCH, name))
        vecs[name] = read_samples(p)
        os.remove(p)
        print(f"{name}: {len(vecs[name])} docs, "
              f"acc {np.mean(list(vecs[name].values())):.4f}")
    common = sorted(set(vecs["mistral7b"]) & set(vecs["qwen15_7b"]))
    a = np.array([vecs["mistral7b"][d] for d in common], dtype=np.int8)
    b = np.array([vecs["qwen15_7b"][d] for d in common], dtype=np.int8)
    np.savez_compressed(os.path.join(HERE, "pair_mmlupro.npz"),
                        doc_id=np.array(common, dtype=np.int32),
                        a=a, b=b,
                        names=np.array(["Mistral-7B-v0.1",
                                        "Qwen1.5-7B"]))
    n = len(common)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    print(f"aligned {n} items; acc A {a.mean():.4f} B {b.mean():.4f}; "
          f"discordant A>B {n10}, B>A {n01}")
