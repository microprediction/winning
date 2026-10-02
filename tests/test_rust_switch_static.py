"""Source-level guards on the compiled-kernel switch, with NO fastrace
dependency.

These lived in test_rust_dispatch_core.py behind a module-level
`pytest.importorskip("fastrace")`, and the required CI matrix installs
`.[test]` without fastrace -- so the guard the docs cite ("fails if any
module imports fastrace off the list") was skipped on every job (#72).
They read the source and spawn a pure-mode interpreter; neither needs the
extension, so they run everywhere.
"""
import pathlib
import re

import winning
from winning import rustconfig


def test_every_module_that_imports_fastrace_is_on_the_switch():
    """The guard that would have caught core. A module that imports
    fastrace but is not registered has a private switch nobody can flip:
    `use_rust(False)` and WINNING_PURE do not reach it, and rust_active()
    lies about it."""
    root = pathlib.Path(winning.__file__).parent
    importers = set()
    for path in root.rglob("*.py"):
        if "research" in path.parts:
            continue
        text = path.read_text()
        if (re.search(r"^\s*import fastrace\b", text, re.M)
                or re.search(r"\bload_fastrace\(", text)):
            rel = path.relative_to(root.parent).with_suffix("")
            importers.add(".".join(rel.parts))
    importers.discard("winning.rustconfig")       # defines the loader
    registered = {m.__name__ for m in rustconfig._rust_modules()}
    missing = sorted(importers - registered)
    assert not missing, (
        "modules that import fastrace but are not in "
        "rustconfig._rust_modules(), so use_rust()/WINNING_PURE cannot "
        "reach them:\n  " + "\n  ".join(missing))


def test_no_module_imports_fastrace_directly():
    """#113: WINNING_PURE has to be decided BEFORE the extension loads,
    or a broken wheel takes the package down under pure mode. Every
    module goes through rustconfig.load_fastrace, which checks first."""
    root = pathlib.Path(winning.__file__).parent
    direct = sorted(
        str(path.relative_to(root.parent))
        for path in root.rglob("*.py")
        if "research" not in path.parts and path.name != "rustconfig.py"
        and re.search(r"^\s*import fastrace\b", path.read_text(), re.M))
    assert not direct, "import fastrace outside load_fastrace:\n  " + "\n  ".join(direct)


def test_pure_mode_never_imports_the_extension():
    """#113: under WINNING_PURE=1 `import winning` must not execute the
    fastrace loader at all (previously it imported, then ignored it)."""
    import subprocess, sys
    root = pathlib.Path(winning.__file__).parent
    code = ("import sys, winning, winning.factor, winning.classic, "
            "winning.methods, winning.alternatives.reprs; "
            "print(winning.rust_active(), 'fastrace' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True,
                         env={**__import__("os").environ, "WINNING_PURE": "1"},
                         cwd=str(root.parent)).stdout.split()
    assert out == ["False", "False"], out
