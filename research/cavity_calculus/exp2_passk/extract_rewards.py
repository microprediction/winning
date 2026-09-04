"""Stream the Pass8-Rollouts JSONL from the Hub and keep only the
outcome table: (prompt index, sample_id, reward, environment). The
3 GB of response text never touches disk; the result is a small npz
committed alongside the experiment."""
import json
import os

import numpy as np
import requests

URL = ("https://huggingface.co/datasets/CL-From-Nothing/"
       "RLVE-Qwen3-4B-Thinking-2507-Pass8-Rollouts/resolve/main/"
       "rlve_train_pass8_0_9000.jsonl")
OUT = os.path.join(os.path.dirname(__file__), "rewards.npz")

idx, sid, rew, env = [], [], [], []
envmap = {}
with requests.get(URL, stream=True, timeout=120) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=False,
                             chunk_size=1 << 20):
        if not line:
            continue
        d = json.loads(line)
        idx.append(d["index"])
        sid.append(d["sample_id"])
        rew.append(d["rewards"])
        e = json.loads(d["metadata"]).get("environment", "?")
        env.append(envmap.setdefault(e, len(envmap)))

env_names = [None] * len(envmap)
for k, v in envmap.items():
    env_names[v] = k
np.savez_compressed(OUT,
                    index=np.array(idx, dtype=np.int32),
                    sample_id=np.array(sid, dtype=np.int8),
                    reward=np.array(rew, dtype=np.float32),
                    env=np.array(env, dtype=np.int16),
                    env_names=np.array(env_names))
print(f"rows {len(idx)}, prompts {len(set(idx))}, "
      f"environments {len(envmap)}; wrote {OUT} "
      f"({os.path.getsize(OUT) / 1e3:.0f} KB)")
