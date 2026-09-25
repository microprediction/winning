"""No port may tabulate the primes behind its low-discrepancy nodes.

A Halton construction needs one prime per dimension. Writing them as a
literal puts a SILENT cliff in a public rank argument: one dimension past
the end of the table, the base is `NA`, `undefined`, or out of bounds,
and the generator either throws somewhere unrelated or -- worse -- fills
that column with nothing and returns numbers that look like an answer.

It has been three separate issues already:

* #190 -- `r/winning`, 30 entries, the public `cov=` route stopped at 31
  runners and threw;
* #233 -- the browser, 16 entries in `demo.mjs` and 24 in `races.mjs`, so
  rank 17 and rank 25 returned NaN for every runner in silence;
* #143 -- `r/mvtnormfast`, 6 entries, explicit rank 7 hit `primes[7] = NA`.

Each was fixed where it was found, and the next one was found somewhere
else, because nobody grepped the other trees. A sweep is the only fix
that generalises, so this test IS the sweep: every port generates its
primes now, and a literal table anywhere fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The opening run of any prime table long enough to be one. Two entries
# could be anything; four in ascending prime order is a table.
TABLE = re.compile(r"\b2\s*,\s*3\s*,\s*5\s*,\s*7\b")

SUFFIXES = {".py", ".R", ".r", ".jl", ".mjs", ".js"}
SKIP_DIRS = {".git", "node_modules", "build", "dist", "__pycache__",
             ".venv", "venv", ".mypy_cache", ".pytest_cache", "attic"}

# The sweep governs SHIPPED SOURCE. A test may state the expected primes
# -- that is how you check a generator -- so test files are exempt, by a
# rule rather than a hand-kept list, since every port spells its test
# paths differently: tests/ and testthat/ directories, `test_*.py`,
# `test-*.R`, `runtests.jl`.
def _is_test_file(rel: str) -> bool:
    parts = rel.split("/")
    if any(part in ("tests", "test", "testthat") for part in parts[:-1]):
        return True
    name = parts[-1]
    return (name.startswith(("test_", "test-", "test."))
            or name == "runtests.jl"
            or name.endswith(("_test.py", "_test.jl", "_test.R")))


def _sources():
    for path in ROOT.rglob("*"):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        yield path


def test_the_sweep_reaches_every_port():
    """A sweep that reads nothing passes silently, which is the exact
    failure mode this file exists to prevent."""
    seen = {p.relative_to(ROOT).as_posix() for p in _sources()}
    assert len(seen) > 100, f"only {len(seen)} source files swept"
    for port in ("winning/factor/races.py", "r/winning/R/races.R",
                 "julia/winning/src/winning.jl", "docs/js/winning/races.mjs",
                 "r/mvtnormfast/R/pmvnorm_fast.R"):
        assert port in seen, f"{port} was not swept"


def test_no_port_tabulates_its_halton_primes():
    offenders = []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        if _is_test_file(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for num, line in enumerate(text.splitlines(), 1):
            if TABLE.search(line):
                offenders.append(f"{rel}:{num}: {line.strip()[:64]}")
    assert not offenders, (
        "a tabulated prime list puts a silent cliff in a public rank "
        "argument; generate them instead (see .first_primes / "
        "_first_primes / firstPrimes):\n  " + "\n  ".join(offenders))


def test_every_port_has_a_prime_generator():
    """The other half: the sweep above passes if a port simply deletes
    its Halton nodes. Each port that has them must have a generator."""
    generators = {
        "r/winning/R/ghk.R": ".first_primes <- function",
        "r/mvtnormfast/R/pmvnorm_fast.R": ".first_primes <- function",
        "r/rprobitfast/R/engine.R": ".first_primes <- function",
        "r/mlogitfast/R/mlogit_fast.R": ".first_primes <- function",
        "julia/winning/src/winning.jl": "function _first_primes",
        "julia/MultinomialProbit/src/MultinomialProbit.jl":
            "function _first_primes",
        "docs/js/winning/core.mjs": "export function firstPrimes",
    }
    for rel, marker in generators.items():
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; update this list"
        assert marker in path.read_text(encoding="utf-8"), (
            f"{rel} no longer defines its prime generator ({marker})")
