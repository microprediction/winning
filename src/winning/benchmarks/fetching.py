"""Shared download-and-cache plumbing for benchmark datasets."""

from __future__ import annotations

import os
import urllib.request


def cache_dir() -> str:
    d = os.environ.get(
        "WINNING_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "winning")
    )
    os.makedirs(d, exist_ok=True)
    return d


def fetch(url: str, filename: str, timeout: int = 120) -> str:
    """Download url to the cache (once) and return the local path."""
    path = os.path.join(cache_dir(), filename)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "winning-benchmarks"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return path
