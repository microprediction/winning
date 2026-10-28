"""Stream the Pass8-Rollouts JSONL once and keep per-sample features
plus the per-prompt 8x8 text-similarity matrix. The response text
never touches disk.

Per sample: index, sample_id, reward, nchar, nwords, think_closed
(contains '</think>'), ans_hash (blake2b of the extracted final
answer, 0 when none), ans_ok (extracted answer == reference answer).
Per prompt: jac[p, i, j] = Jaccard similarity of word 3-shingle sets
of responses i and j; prefix_len = length of the common prefix of all
eight responses (a fixed opener is expected).
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import requests

URL = ("https://huggingface.co/datasets/CL-From-Nothing/"
       "RLVE-Qwen3-4B-Thinking-2507-Pass8-Rollouts/resolve/main/"
       "rlve_train_pass8_0_9000.jsonl")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "features.npz")
G = 8
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
WORD = re.compile(r"\w+")


def final_answer(resp):
    tail = resp.rsplit("</think>", 1)[1] if "</think>" in resp else ""
    m = BOXED.findall(tail) or BOXED.findall(resp[-4000:])
    if m:
        return m[-1].strip()
    tail = tail.strip()
    return tail.splitlines()[-1].strip()[:200] if tail else ""


def shingles(resp):
    w = WORD.findall(resp.lower())
    return set(zip(w, w[1:], w[2:]))


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8)
                          .digest(), "little", signed=True) if s else 0


rows = {k: [] for k in ("index", "sample_id", "reward", "nchar",
                        "nwords", "think_closed", "ans_hash", "ans_ok")}
jac, prefix_len, prompt_index = [], [], []
group, gsets = [], []
t0 = time.time()


def flush():
    if not group:
        return
    n = len(group)
    J = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gsets[i], gsets[j]
            u = len(a | b)
            J[i, j] = J[j, i] = len(a & b) / u if u else 0.0
    jac.append(J)
    prefix_len.append(len(os.path.commonprefix(group)))
    group.clear()
    gsets.clear()


cur = None
with requests.get(URL, stream=True, timeout=300) as r:
    r.raise_for_status()
    for k, line in enumerate(r.iter_lines(chunk_size=1 << 22)):
        if not line:
            continue
        d = json.loads(line)
        if d["index"] != cur:
            flush()
            cur = d["index"]
            prompt_index.append(cur)
        resp = d["response"]
        ans = final_answer(resp)
        rows["index"].append(d["index"])
        rows["sample_id"].append(d["sample_id"])
        rows["reward"].append(d["rewards"])
        rows["nchar"].append(len(resp))
        rows["nwords"].append(len(WORD.findall(resp)))
        rows["think_closed"].append("</think>" in resp)
        rows["ans_hash"].append(h64(ans))
        rows["ans_ok"].append(ans == str(d["answer"]).strip())
        group.append(resp[:2000])
        gsets.append(shingles(resp))
        if k % 4000 == 0:
            el = time.time() - t0
            print(f"{k:,} rows  {el:.0f}s  {k / max(el, 1e-9):.0f} rows/s",
                  file=sys.stderr, flush=True)
flush()

assert all(len(J) == G for J in jac), "a prompt without 8 samples"
np.savez_compressed(
    OUT,
    prompt_index=np.array(prompt_index, dtype=np.int32),
    jac=np.stack(jac),
    prefix_len=np.array(prefix_len, dtype=np.int32),
    index=np.array(rows["index"], dtype=np.int32),
    sample_id=np.array(rows["sample_id"], dtype=np.int8),
    reward=np.array(rows["reward"], dtype=np.float32),
    nchar=np.array(rows["nchar"], dtype=np.int32),
    nwords=np.array(rows["nwords"], dtype=np.int32),
    think_closed=np.array(rows["think_closed"], dtype=bool),
    ans_hash=np.array(rows["ans_hash"], dtype=np.int64),
    ans_ok=np.array(rows["ans_ok"], dtype=bool),
)
print(f"rows {len(rows['index']):,}, prompts {len(jac):,}; wrote {OUT} "
      f"({os.path.getsize(OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
