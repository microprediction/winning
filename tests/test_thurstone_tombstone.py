"""``winning.thurstone`` fails honestly: an ImportError that says where
the code went and what to use, not a bare 'No module named'."""
import importlib
import subprocess
import sys

import pytest


def test_importing_the_tombstone_raises_a_useful_error():
    sys.modules.pop("winning.thurstone", None)
    with pytest.raises(ImportError) as e:
        importlib.import_module("winning.thurstone")
    msg = str(e.value)
    assert "winning.thurstone was removed" in msg
    assert "from winning.research import" in msg           # where it went
    assert "calibrate_abilities(p, V=, D=)" in msg          # what to prefer
    assert "import thurstone" in msg                        # the shim route
    assert "No module named" not in msg


def test_from_winning_import_thurstone_gets_the_same_message():
    sys.modules.pop("winning.thurstone", None)
    with pytest.raises(ImportError, match=r"winning\.thurstone was removed"):
        from winning import thurstone  # noqa: F401


def test_importing_winning_itself_is_unaffected():
    """The tombstone must not be imported eagerly by the package."""
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c",
         "import winning, winning.research; print('ok', winning.rust_active())"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")
