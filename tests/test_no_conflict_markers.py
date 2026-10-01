"""No unresolved merge conflict lands in a tracked file.

`parity/check_js_api.mjs` sat on main with a three-line conflict in it.
The file did not PARSE, so `node parity/check_js_api.mjs` -- ninety-three
behavioural checks of the browser API guards, the ones that exist
because #186 shipped past source-reading tests -- exited on a
SyntaxError instead of checking anything. Both sides of that conflict
were correct and additive; deleting the three marker lines makes every
check pass. Nothing else had to change, which is exactly why it went
unnoticed: the merge resolution was right and only its leftovers were
committed.

Three things had to line up for that to reach main, and the other two
are not this test's business: CI had not run on main for five weeks
(the runner pool was saturated, and a push at saturation creates no run
at all), and the branch it came from was merged without rerunning the
checker. This test is the cheap one -- a marker is visible in the text
and needs no toolchain to find, so it should never be a toolchain's job.

The marker strings are BUILT here rather than written out, so this file
does not match its own sweep. A previous sweep of mine tripped on its
own fixtures twice.
"""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# assembled so the literals never appear in this file's own bytes
_LT, _EQ, _GT = "<" * 7, "=" * 7, ">" * 7
MARKERS = (_LT + " ", _GT + " ", "|" * 7 + " ")


def _tracked_files(root=ROOT):
    # git is the file list, so a source tree unpacked without it has
    # nothing to sweep rather than a wrong answer. That is a skip, not
    # a pass: a sweep that silently reads nothing passes forever.
    try:
        r = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                           capture_output=True, text=True)
    except (OSError, FileNotFoundError):
        pytest.skip("no git available to list tracked files")
    if r.returncode != 0:
        pytest.skip(f"not a git checkout: {r.stderr.strip()[:60]}")
    return [root / n for n in r.stdout.split("\0") if n]


def _text_lines(path):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None          # a binary blob cannot hold a textual marker


def _offenders(root=ROOT):
    bad = []
    for path in _tracked_files(root):
        lines = _text_lines(path)
        if lines is None:
            continue
        for i, line in enumerate(lines, 1):
            # a marker is at the START of its line, and `=======` alone
            # is also a reStructuredText underline, so it is only an
            # offence next to one of the directional markers
            if line.startswith(MARKERS):
                bad.append((path.relative_to(root), i, line[:40]))
    return bad


def test_no_tracked_file_has_a_conflict_marker():
    bad = _offenders()
    assert not bad, "unresolved merge conflict in:\n" + "\n".join(
        f"  {p}:{i}  {t}" for p, i, t in bad)


def test_the_equals_run_is_only_an_offence_beside_a_real_marker():
    # `=======` opens no conflict on its own: it underlines headings in
    # reStructuredText and rules off sections in markdown. The sweep
    # must not fire on those, or it gets disabled the first time
    # someone writes a document.
    assert _EQ not in MARKERS
    assert not _EQ.startswith(MARKERS)


def test_the_sweep_actually_reads_files(tmp_path):
    # A sweep that silently reads nothing passes forever. Pin that it
    # sees a real, sizeable population of this repo's own text.
    seen = [p for p in _tracked_files() if _text_lines(p) is not None]
    assert len(seen) > 200, f"only {len(seen)} text files swept"


def test_a_planted_marker_is_found(tmp_path):
    # Sabotage: the sweep must fail on a file that has one. Run it
    # against a throwaway git repo rather than dirtying this one.
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("fine\n")
    (repo / "b.js").write_text("ok\n" + _LT + " HEAD\nx\n" + _EQ + "\ny\n"
                               + _GT + " origin/main\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    bad = _offenders(repo)
    # both directional markers are caught, and the clean file is not;
    # the bare ======= on line 4 is deliberately NOT an offence
    assert {str(p) for p, _, _ in bad} == {"b.js"}, bad
    assert sorted(i for _, i, _ in bad) == [2, 6], bad
