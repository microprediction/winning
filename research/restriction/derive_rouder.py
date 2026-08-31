"""Reduce the twelve-line deposit to the counts we actually score, and fetch it if absent.

The upstream repository, `PerceptionCognitionLab/data0` under `1dMemory/chunk`, carries no
licence file, no README and no associated publication, so we do not redistribute it. What is
committed here instead is the derived confusion counts: for each subject and each offered
label set, how often stimulus s drew response r. Those are measurements of the experiment
rather than a copy of the deposit, they are what every figure in the paper rests on, and they
are four orders of magnitude smaller.

  python derive_rouder.py --fetch     clone the upstream deposit into data/rouder_chunk_raw
  python derive_rouder.py             rebuild data/rouder_chunk/counts.npz from it

`rouder_chunk.py` reads counts.npz and never touches the raw files, so the analysis
reproduces from what is committed. Rebuilding from upstream is only needed to verify the
reduction, and MANIFEST.tsv records the upstream git blob SHA of every file it was built from.
"""
import collections
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "rouder_chunk_raw"
OUT = HERE / "data" / "rouder_chunk" / "counts.npz"
UPSTREAM = "https://github.com/PerceptionCognitionLab/data0.git"
K = 12


def fetch():
    """Sparse-clone the one directory we read."""
    if RAW.exists():
        print(f"{RAW} exists already; delete it to refetch")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", UPSTREAM, str(RAW)],
                   check=True)
    subprocess.run(["git", "-C", str(RAW), "sparse-checkout", "set", "1dMemory/chunk"],
                   check=True)
    print(f"fetched into {RAW}")


def read_trials(path):
    out = []
    for line in open(path):
        t = line.split()
        if len(t) != 7:
            continue
        out.append((int(t[1][3:]), int(t[4]), int(t[5])))    # block, stimulus, response
    return out


def build():
    src = RAW / "1dMemory" / "chunk" / "c0"
    if not src.exists():
        sys.exit(f"no raw deposit at {src}; run with --fetch first")
    blocks_out, meta = [], []
    skipped = 0
    for path in sorted(glob.glob(str(src / "C1*S*"))):
        name = Path(path).name
        cond = name[2]
        rows = read_trials(path)
        if not rows:
            continue
        per_block = collections.defaultdict(list)
        for blk, st, rp in rows:
            per_block[blk].append((st, rp))
        for blk, trials in sorted(per_block.items()):
            # some blocks carry codes outside 0..11. rouder_chunk.py drops those blocks,
            # because no design set contains the code, so drop them here too and keep the
            # two paths identical.
            if any(x < 0 or x >= K for st, rp in trials for x in (st, rp)):
                skipped += 1
                continue
            counts = np.zeros((K, K), dtype=np.int32)
            for st, rp in trials:
                counts[st, rp] += 1
            blocks_out.append(counts)
            meta.append((name, cond, blk))
    if not blocks_out:
        sys.exit("no trials parsed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        counts=np.stack(blocks_out),
        subject=np.array([m[0] for m in meta]),
        condition=np.array([m[1] for m in meta]),
        block=np.array([m[2] for m in meta], dtype=np.int32),
    )
    print(f"skipped {skipped} blocks carrying out-of-range codes")
    print(f"wrote {OUT} with {len(blocks_out)} blocks from "
          f"{len({m[0] for m in meta})} subject files, "
          f"{OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
