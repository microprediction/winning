"""The frozen Glicko-2 in attic/src loads and runs.

Six chess experiments compare against it by path, and nothing in the live
package does Glicko-2. It had stopped working without anyone noticing:
`exact.py` imported the retired external `thurstone` package, and
`winning.thurstone` is now a tombstone that raises on import, so every one
of those experiments would have failed at the Glicko-2 step. It imports
`winning.research` now, which is the migration that tombstone prescribes.

This test exists because nothing else looks at that directory -- it is not
packaged and not on pytest's testpaths, which is exactly how it broke.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ATTIC = ROOT / "attic" / "src" / "winning"


@pytest.fixture(scope="module")
def frozen():
    spec = importlib.util.spec_from_file_location(
        "_attic_winning", ATTIC / "__init__.py",
        submodule_search_locations=[str(ATTIC)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_attic_winning"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("_attic_winning", None)


def test_the_package_init_matches_what_is_there():
    """Everything with a live equivalent was deleted; the init must not
    still import it."""
    present = {p.stem for p in ATTIC.glob("*.py")} - {"__init__"}
    assert present == {"exact", "glicko2", "ratingsystem"}, present
    text = (ATTIC / "__init__.py").read_text()
    for gone in ("kernels", "elo", "thurstonerating", "shims"):
        assert gone not in text, f"__init__ still imports the deleted {gone}"


def test_glicko2_loads_and_rates(frozen):
    g = frozen.Glicko2Rating()
    for _ in range(3):
        g.observe(["A", "B"], [1, 2])          # A first, B second
    board = dict((n, r.mu) for n, r in g.leaderboard())
    assert board["A"] > 1500 > board["B"]      # the winner gained
    p = list(g.win_probabilities(["A", "B"]))
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[1]


def test_the_loaders_point_at_the_attic():
    """The six experiments find it by path; if the path rots they fail at
    the Glicko-2 step, which is how this broke the first time."""
    loaders = sorted((ROOT / "research" / "chess").glob("exp*.py"))
    users = [p for p in loaders if "wsrc.glicko2" in p.read_text()]
    assert users, "no chess experiment loads glicko2 any more; drop the attic"
    for p in users:
        text = p.read_text()
        assert '"attic", "src", "winning"' in text or \
               "attic/src/winning" in text, f"{p.name} has a stale path"
