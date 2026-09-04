"""Aligned per-instance GSM outcomes for model pairs from the public
HELM Lite bucket (no auth). Pairs chosen by full-data accuracy gap:
near-tie, two hard-but-decidable, one easy sanity pair."""
import json
import os

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    "mistral7b": ("gsm:model=mistralai_mistral-7b-v0.1", "v1.0.0"),
    "yi6b": ("gsm:model=01-ai_yi-6b", "v1.0.0"),
    "llama3_8b": ("gsm:model=meta_llama-3-8b", "v1.2.0"),
    "llama65b": ("gsm:model=meta_llama-65b", "v1.0.0"),
    "gemma7b": ("gsm:model=google_gemma-7b", "v1.2.0"),
    "llama2_70b": ("gsm:model=meta_llama-2-70b", "v1.0.0"),
    "qwen15_14b": ("gsm:model=qwen_qwen1.5-14b", "v1.2.0"),
}
PAIRS = [("mistral7b", "yi6b"), ("llama3_8b", "llama65b"),
         ("gemma7b", "llama2_70b"), ("qwen15_14b", "gemma7b")]


def per_instance(run, suite):
    url = ("https://storage.googleapis.com/crfm-helm-public/lite/"
           f"benchmark_output/runs/{suite}/{run}/per_instance_stats.json")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out = {}
    for rec in r.json():
        for st in rec["stats"]:
            if st["name"]["name"] in ("final_number_exact_match",
                                      "exact_match"):
                out[rec["instance_id"]] = int(round(st.get("mean", 0)))
                break
    return out


if __name__ == "__main__":
    vecs = {k: per_instance(*v) for k, v in RUNS.items()}
    for name, d in vecs.items():
        print(name, len(d), round(sum(d.values()) / len(d), 4))
    for a_name, b_name in PAIRS:
        common = sorted(set(vecs[a_name]) & set(vecs[b_name]))
        a = np.array([vecs[a_name][i] for i in common], dtype=np.int8)
        b = np.array([vecs[b_name][i] for i in common], dtype=np.int8)
        np.savez_compressed(
            os.path.join(HERE, f"pair_{a_name}_vs_{b_name}.npz"),
            a=a, b=b, names=np.array([a_name, b_name]))
        print(f"{a_name} vs {b_name}: {len(common)} aligned, "
              f"acc {a.mean():.3f}/{b.mean():.3f}, discordant "
              f"{int(((a==1)&(b==0)).sum())}/{int(((a==0)&(b==1)).sum())}")
