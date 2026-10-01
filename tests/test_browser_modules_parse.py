"""Every javascript module in the repo parses.

`parity/check_js_api.mjs` sat on main with an unresolved merge conflict
in it, so `node parity/check_js_api.mjs` exited on a SyntaxError and its
ninety-odd behavioural checks never ran. #275 removed those markers and
added `tests/test_no_conflict_markers.py`, which finds a marker in any
tracked text file without needing a toolchain.

That is not quite enough, and I know because I hit the gap immediately
after. Resolving a later conflict in the same file by keeping both
sides dropped ONE closing brace -- no markers left, marker sweep green,
file would not parse. A conflict resolution is exactly the moment a
brace goes missing, and the marker sweep cannot see it.

So: ask node. This needs a toolchain and is therefore skipped where
there is none, which is why it is the second net and not the first.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _tracked_js():
    r = subprocess.run(["git", "ls-files", "-z", "*.mjs", "*.js"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return sorted(ROOT / n for n in r.stdout.split("\0") if n)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_every_tracked_javascript_module_parses():
    bad = []
    for path in _tracked_js():
        # --check reads the file and reports syntax only; it does not
        # execute it, so a module with side effects is safe here
        flag = "--check"
        r = subprocess.run([NODE, flag, str(path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            first = next((ln for ln in r.stderr.splitlines()
                          if "Error" in ln), r.stderr.strip()[:80])
            bad.append(f"  {path.relative_to(ROOT)}: {first}")
    assert not bad, "javascript that does not parse:\n" + "\n".join(bad)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_sweep_actually_reads_the_browser_engine():
    """A sweep that silently matches nothing passes forever. The browser
    engine is the thing this most needs to cover, so require it."""
    found = {p.relative_to(ROOT).as_posix() for p in _tracked_js()}
    assert "parity/check_js_api.mjs" in found
    assert any(p.startswith("docs/js/winning/") for p in found)
    assert len(found) >= 10, sorted(found)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_missing_brace_is_caught(tmp_path):
    """Sabotage, on the exact shape that got past the marker sweep: a
    resolution that keeps both sides and loses one closing brace."""
    good = tmp_path / "good.mjs"
    good.write_text("export function f() {\n  return 1;\n}\n")
    bad = tmp_path / "bad.mjs"
    bad.write_text("export function f() {\n  return 1;\n")   # no closing }
    assert subprocess.run([NODE, "--check", str(good)]).returncode == 0
    assert subprocess.run([NODE, "--check", str(bad)],
                          capture_output=True).returncode != 0
