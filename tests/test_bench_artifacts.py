"""The benchmark entry points work from an installed artifact (#141).

Three things failed from a clean wheel/sdist: `winning[benchmarks]`
installed pandas only, so the TrueSkill comparators died at import (or
with an AttributeError on None); `python -m winning.bench.runner` wrote to
`Path(__file__).parents[2] / "bench_results"`, which from a wheel is
<site-packages>/bench_results; and the sdist shipped
tests/test_golden_and_convergence.py without tests/golden/race_boards.npz.
The installed-wheel and extracted-sdist halves run in ci.yml's `package`
job; these are the source-level guards.
"""
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "winning" / "bench"


def test_no_bench_script_writes_next_to_the_package():
    offenders = [f.name for f in BENCH.glob("*.py")
                 if re.search(r"__file__\)?\.resolve\(\)\.parents", f.read_text())]
    assert not offenders, offenders


def test_results_dir_defaults_to_cwd(tmp_path, monkeypatch):
    from winning.bench import results_dir
    monkeypatch.delenv("WINNING_BENCH_RESULTS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert results_dir() == (tmp_path / "bench_results").resolve()
    assert results_dir(tmp_path / "x") == (tmp_path / "x").resolve()
    monkeypatch.setenv("WINNING_BENCH_RESULTS", str(tmp_path / "env"))
    assert results_dir() == (tmp_path / "env").resolve()


def test_leaderboard_reads_and_writes_out(tmp_path):
    from winning.bench import leaderboard
    rec = {"problem": "p", "n": 2, "k": 1, "method": "m", "budget": None,
           "seconds": 0.1, "max_err": 1e-4, "max_log_err": 0.01, "info": {}}
    (tmp_path / "records.jsonl").write_text(json.dumps(rec) + "\n")
    leaderboard.main(out=tmp_path)
    assert (tmp_path / "LEADERBOARD.md").is_file()


@pytest.mark.parametrize("mod", ["winning.bench.season_ranked",
                                 "winning.bench.season_trueskill"])
def test_trueskill_benchmarks_import_without_it_and_say_how_to_get_it(mod, monkeypatch):
    monkeypatch.setitem(sys.modules, "trueskill", None)   # import -> ImportError
    for name in [mod, "winning.bench"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    m = importlib.import_module(mod)
    with pytest.raises(ImportError, match=r"winning\[benchmarks\]"):
        if hasattr(m, "season"):
            m.season(1.0, P=10, field=3, races=2)
        else:
            m.main(P=10, field=3, races=2)


def test_benchmarks_extra_declares_the_comparators():
    text = (ROOT / "setup.py").read_text()
    m = re.search(r'"benchmarks":\s*\[([^\]]*)\]', text)
    assert m, "setup.py lost its benchmarks extra"
    extra = set(re.findall(r'"([A-Za-z0-9_.-]+)', m.group(1)))
    imported = set()
    for f in BENCH.glob("*.py"):
        imported |= set(re.findall(r"^\s*import (trueskill|openskill|pandas)\b",
                                   f.read_text(), re.M))
    assert imported - {"pandas"} <= extra, (imported, extra)
    assert "trueskill" in extra


def test_sdist_does_not_ship_the_repository_test_suite():
    """The suite reads parity/, r/, docs/js/, golden data and the committed
    results; an sdist without them cannot run it."""
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert re.search(r"(?m)^prune tests\s*$", manifest)
