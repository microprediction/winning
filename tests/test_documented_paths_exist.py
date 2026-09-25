"""A backticked in-repo path is a promise that it is there.

PR #220 deleted the dead `src/` package, and two documents kept pointing
into it: `data/README.md` sent readers to a benchmarks tree that no
longer exists, and a research script explained that `pip install winning`
resolves to the attic, which `setup.py` shows it never did (#250).

The convention this pins is narrow on purpose. A repo-wide "every path in
every document exists" check is not worth having here: of 216 backticked
paths in markdown, 115 do not resolve, and nearly all of those are URLs,
MIME types, external dataset identifiers, or paths relative to the
document's own directory. Blessing a hundred exceptions would cost more
than it catches.

So the rule is about the tree that just moved: a path under `attic/`
written in BACKTICKS must exist. Backticks mean "go and look"; prose
about something that used to be there is fine and is how the deleted
benchmarks tree is now described.

The second half pins the claim the research script makes about
packaging, because that is the sort of statement that goes quietly
false.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv"}


def _documents():
    for pattern in ("*.md", "*.py"):
        for f in ROOT.rglob(pattern):
            rel = f.relative_to(ROOT)
            if SKIP & set(rel.parts):
                continue
            if rel.parts and rel.parts[0] == "attic":
                continue          # the attic may describe its own past
            if f.resolve() == Path(__file__).resolve():
                continue          # this file quotes the patterns it hunts
            yield f


def test_the_sweep_reads_the_documents():
    """An empty sweep would pass the assertion below in silence."""
    found = list(_documents())
    assert len(found) > 100, f"only swept {len(found)} documents"
    names = {f.relative_to(ROOT).as_posix() for f in found}
    assert "data/README.md" in names
    assert "README.md" in names


def test_a_backticked_attic_path_exists():
    offenders = []
    for f in _documents():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for num, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r"`(attic/[\w./-]+)`", line):
                path = m.group(1).rstrip("/")
                if not (ROOT / path).exists():
                    offenders.append(
                        f"{f.relative_to(ROOT).as_posix()}:{num}: {path}")
    assert not offenders, (
        "documents pointing into the attic at paths that are not there. "
        "Backticks mean the reader can go and look; describe something "
        "that USED to be there in prose instead:\n  "
        + "\n  ".join(offenders))


def test_setup_packages_the_top_level_tree_not_the_attic():
    """`research/polysemy_pilot/exact_analyze.py` explains which tree an
    install resolves to. It said the attic, which was never true."""
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "attic" not in source, "setup.py now mentions the attic"
    tree = ast.parse(source)
    packages = None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "packages":
            if isinstance(node.value, (ast.List, ast.Tuple)):
                packages = [e.value for e in node.value.elts
                            if isinstance(e, ast.Constant)]
    assert packages, "setup.py no longer lists its packages literally"
    assert "winning" in packages
    assert all(p == "winning" or p.startswith("winning.") for p in packages), (
        f"setup.py packages something outside the top-level tree: {packages}")


def test_no_document_claims_an_install_resolves_to_the_attic():
    offenders = []
    for f in _documents():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for num, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if "pip install winning" in low and "attic" in low:
                offenders.append(
                    f"{f.relative_to(ROOT).as_posix()}:{num}")
    assert not offenders, (
        "a document says an install of `winning` resolves to the attic; "
        "setup.py packages the top-level tree:\n  " + "\n  ".join(offenders))
