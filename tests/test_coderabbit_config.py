"""Every maintained port gets CodeRabbit's parity instruction (#431).

CodeRabbit applies a path instruction only to files matching its glob, so
a port-only PR (say one touching just docs/js/winning/structures.mjs) got
the generic review and never the instruction to compare with the Python
reference. That is the direction in which the #322 R/browser gap reached
main. This test fails if a port root, or any file under one, is left out.

Parsed without PyYAML (not in the `test` extra): the path_instructions
block is a flat list of `- path:` / `instructions: >-` pairs. When PyYAML
is present the hand parse is cross-checked against it.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".coderabbit.yaml"

# Every directory holding a port of the Python reference.
PORT_ROOTS = ["r", "js", "docs/js/winning", "julia", "rust"]

# Build outputs that are never committed or reviewed.
SKIP_PARTS = {"target", "node_modules", ".git", "__pycache__"}


def _glob_to_regex(glob):
    """minimatch-style: `**` spans directories, `*` stays within one."""
    out, i = "", 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif glob.startswith("**", i):
            out += ".*"
            i += 2
        elif glob[i] == "*":
            out += "[^/]*"
            i += 1
        else:
            out += re.escape(glob[i])
            i += 1
    return re.compile(out + r"\Z")


def _parse(text):
    """Return (path_filters, [(path, instructions), ...])."""
    filters, instructions = [], []
    section = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if re.match(r"^  \w", line):
            section = stripped.rstrip(":")
        if section == "path_filters":
            m = re.match(r'^\s+-\s+"(.*)"\s*$', line)
            if m:
                filters.append(m.group(1))
        elif section == "path_instructions":
            m = re.match(r'^\s+-\s+path:\s+"(.*)"\s*$', line)
            if m:
                path = m.group(1)
                i += 1
                assert lines[i].strip() == "instructions: >-", lines[i]
                body = []
                i += 1
                while i < len(lines) and lines[i].startswith(" " * 8):
                    body.append(lines[i].strip())
                    i += 1
                instructions.append((path, " ".join(body)))
                continue
        i += 1
    return filters, instructions


FILTERS, INSTRUCTIONS = _parse(CONFIG.read_text())


def _excluded(rel):
    return any(f.startswith("!") and _glob_to_regex(f[1:]).match(rel)
               for f in FILTERS)


def _parity_globs():
    """Globs whose instruction asks for comparison with the Python
    reference and names a parity checker."""
    return [p for p, text in INSTRUCTIONS
            if "Python" in text and "parity" in text]


def test_hand_parse_matches_yaml():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG.read_text())["reviews"]
    assert FILTERS == cfg["path_filters"]
    assert [(d["path"], d["instructions"].strip())
            for d in cfg["path_instructions"]] == INSTRUCTIONS


@pytest.mark.parametrize("root", PORT_ROOTS)
def test_every_port_root_has_a_parity_instruction(root):
    probe = f"{root}/some/new_file.txt"
    hits = [g for g in _parity_globs() if _glob_to_regex(g).match(probe)]
    assert hits, f"no parity path_instruction covers {root}/**"


def test_every_reviewed_port_file_is_covered():
    globs = [_glob_to_regex(g) for g in _parity_globs()]
    missed = []
    for root in PORT_ROOTS:
        for f in (ROOT / root).rglob("*"):
            if not f.is_file() or SKIP_PARTS & set(f.relative_to(ROOT).parts):
                continue
            rel = f.relative_to(ROOT).as_posix()
            if not _excluded(rel) and not any(g.match(rel) for g in globs):
                missed.append(rel)
    assert not missed, missed[:20]


def test_python_reference_points_at_every_port():
    text = dict(INSTRUCTIONS)["winning/**/*.py"]
    for name in ("r/winning", "js/", "docs/js/winning", "julia/", "rust/"):
        assert name in text, name


def test_glob_matcher():
    assert _glob_to_regex("r/**").match("r/winning/R/races.R")
    assert _glob_to_regex("winning/**/*.py").match("winning/races.py")
    assert not _glob_to_regex("winning/**/*.py").match("docs/js/winning/a.mjs")
    assert _glob_to_regex("**/*.csv").match("a/b/c.csv")
    assert not _glob_to_regex("r/**").match("rust/fastrace/src/lib.rs")
