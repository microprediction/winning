"""The parity scripts import winning from their own checkout (#460)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARITY = os.path.join(ROOT, "parity")


def test_gen_vectors_pins_this_checkout(tmp_path):
    # a conflicting `winning` earlier on the path, as an editable install
    # from another worktree is: run as a script, parity/ is sys.path[0]
    fake = tmp_path / "winning"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise ImportError('wrong checkout')\n")
    code = ("import sys; sys.path[0] = %r; import gen_vectors, winning; print(winning.__file__)" % PARITY)
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert os.path.abspath(out.stdout.strip()).startswith(os.path.join(ROOT, "winning"))
