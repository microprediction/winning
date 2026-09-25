"""A research script that promises the checkout must deliver it.

`exact_analyze.py` documents that the repository root "goes ahead of
site-packages" so the script "always reads the tree it sits in". It
inserted the root only when the root was ABSENT, which is not the same
thing: with `pip install -e` plus a plain `python script.py` the root is
already on sys.path, behind site-packages, and the import took the
installed copy while the comment said otherwise (#305).

These run a subprocess with a DECOY `winning` package ahead of the root,
because the only convincing test of "which copy gets imported" is to
provide two and see which one wins.
"""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "research", "polysemy_pilot", "exact_analyze.py")


@pytest.fixture
def decoy(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    """A `winning.factor.core` that is importable and clearly not ours."""
    pkg = tmp_path / "winning" / "factor"
    pkg.mkdir(parents=True)
    (tmp_path / "winning" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(
        "DECOY = True\n"
        "def win_probabilities(*a, **k): raise AssertionError('decoy')\n"
        "def abilities_from_probabilities(*a, **k): "
        "raise AssertionError('decoy')\n")
    return str(tmp_path)


def _run(code, decoy_dir, cwd):
    """Run with the decoy FIRST on sys.path and the repo root after it,
    which is the ordering the old guard could not correct.

    From a NEUTRAL cwd: run from the repo root, `python -c` puts the
    checkout at sys.path[0] itself and the decoy never wins, so the bug
    is invisible exactly where a developer would look for it. The
    comment promises the checkout "whatever the caller's cwd", so the
    test has to leave."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([decoy_dir, ROOT])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, env=env,
                          cwd=str(cwd))


def test_the_decoy_really_would_win_without_the_guard(decoy, tmp_path):
    """Guard the guard: if the decoy did not shadow, the next test would
    pass no matter what the script did."""
    out = _run("""
        import winning.factor.core as c
        print(getattr(c, "DECOY", False))
    """, decoy, tmp_path / "elsewhere")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True", out.stdout


def test_the_script_reads_the_tree_it_sits_in(decoy, tmp_path):
    out = _run(f"""
        import sys, types
        # exact_restrict builds an OpenAI client at import time; stub it
        # so this exercises the real path prologue and nothing else
        _s = types.ModuleType("exact_restrict")
        _s.CELLS = []; _s.MODELS = []; _s.PHRASINGS = []
        _s.match_items = lambda *a, **k: None
        sys.modules["exact_restrict"] = _s
        import importlib.util
        spec = importlib.util.spec_from_file_location("ea", {SCRIPT!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import winning.factor.core as c
        print(getattr(c, "DECOY", False), c.__file__)
    """, decoy, tmp_path / "elsewhere")
    assert out.returncode == 0, out.stderr
    flag, path = out.stdout.strip().split(" ", 1)
    assert flag == "False", "the script imported the decoy"
    assert os.path.abspath(path).startswith(os.path.abspath(ROOT)), path


def test_the_root_is_first_and_appears_once(decoy, tmp_path):
    out = _run(f"""
        import sys, types
        # exact_restrict builds an OpenAI client at import time; stub it
        # so this exercises the real path prologue and nothing else
        _s = types.ModuleType("exact_restrict")
        _s.CELLS = []; _s.MODELS = []; _s.PHRASINGS = []
        _s.match_items = lambda *a, **k: None
        sys.modules["exact_restrict"] = _s
        import importlib.util
        spec = importlib.util.spec_from_file_location("ea", {SCRIPT!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        root = {ROOT!r}
        print(sys.path.index(root), sys.path.count(root))
    """, decoy, tmp_path / "elsewhere")
    assert out.returncode == 0, out.stderr
    index, count = out.stdout.strip().split()
    assert index == "0", f"repo root is at index {index}, not first"
    assert count == "1", f"repo root appears {count} times"
