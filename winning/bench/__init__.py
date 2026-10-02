"""Benchmark harness: seeded problem grid, cached references, append-only
accuracy-time records. Excluded from wheels' runtime path concerns by
having no heavy imports; contestants come from winning.methods.

Output goes to ./bench_results under the CURRENT directory (or --out / the
WINNING_BENCH_RESULTS environment variable), never next to the installed
package: from a wheel that was <venv>/site-packages/bench_results, which
polluted a writable install and failed on a read-only one (#141). Run from
the repository root to keep using the committed bench_results/."""

import os
from pathlib import Path


def results_dir(out=None):
    """Where the benchmark scripts read and write: `out` if given, else
    $WINNING_BENCH_RESULTS, else ./bench_results."""
    return Path(out or os.environ.get("WINNING_BENCH_RESULTS") or "bench_results").resolve()


def require_trueskill():
    """The reference TrueSkill comparator, with an install hint instead of
    an AttributeError on None."""
    try:
        import trueskill
    except ImportError as e:
        raise ImportError("this benchmark compares against TrueSkill; "
                          "pip install 'winning[benchmarks]' (or trueskill)") from e
    return trueskill
