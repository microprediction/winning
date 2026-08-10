"""Crash-safe persistence for API-derived experiment data.

Rule for every battery in this directory: a paid API response is written to
an append-only JSONL log the moment it arrives, before anything is computed
from it. Derived scores are recomputable from that log forever; the log is
the only irreplaceable artifact, so nothing may truncate or overwrite it.

- `append_jsonl` appends one record, flushes and fsyncs, under a lock so
  worker threads cannot interleave partial lines. A run killed at any point
  keeps every response it already paid for.
- `load_jsonl` tolerates a torn final line (possible only if the process died
  inside the write) and skips it rather than failing the whole load.
- `write_atomic` writes consolidated/derived files via temp-file + rename, so
  a crash mid-write leaves the previous good version in place instead of a
  truncated one.
"""
import json
import os
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()


def append_jsonl(path, record):
    """Append one JSON record durably. Safe to call from many threads."""
    line = json.dumps(record) + "\n"
    with _LOCK:
        with open(path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def load_jsonl(path, key=None):
    """Load records. With `key`, return a dict keyed by record[key] (last
    write wins). Ignores a torn trailing line from a killed process."""
    p = Path(path)
    if not p.exists():
        return {} if key else []
    out = {} if key else []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line
        if key:
            out[rec[key]] = rec
        else:
            out.append(rec)
    return out


def write_atomic(path, text):
    """Replace `path` only once the new contents are fully on disk."""
    p = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path, obj, indent=1):
    write_atomic(path, json.dumps(obj, indent=indent))
