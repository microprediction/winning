"""The ratings verifier as a CI gate (winning.ratings.verify).

The FAST profile must be clean on every push: no FAIL and no
UNDERPOWERED (EXPECTED_APPROX rows are allowed and printed). The
registry must be well formed (unique names, every non-MEASURED result
backed by a mark, JSON round trip), seeds must not depend on execution
order or worker count, and the exit-code semantics must hold: a thin
cell is never a pass.
"""
import json
import os
import time

import numpy as np
import pytest

from winning.ratings.verify import (CHECKS, PROFILES, Report, Result,
                                    seed_for, verify)
from winning.ratings.verify import marks

FAST_BUDGET_S = 60.0


def test_fast_profile_is_clean():
    t0 = time.time()
    rep = verify(profile="fast", verbose=False)
    wall = time.time() - t0
    assert rep.results, "no checks ran"
    assert not rep.failed and not rep.underpowered, rep.to_markdown()
    assert rep.exit_code == 0
    if os.environ.get("CI"):
        assert wall < 4 * FAST_BUDGET_S, f"fast profile took {wall:.0f}s"


def test_registry_is_well_formed():
    assert CHECKS, "no checks registered"
    names = list(CHECKS)
    assert len(names) == len(set(names))
    for n, c in CHECKS.items():
        assert c.profiles <= set(PROFILES)
        assert "exhaustive" in c.profiles          # every check runs somewhere
        assert c.cost_s > 0
    for name, m in marks.MARKS.items():
        assert {"set_by", "date", "basis"} <= set(m), name
        assert ("tolerance" in m) or ("tolerance_by_n" in m) or ("thresholds" in m), name


def test_results_round_trip_through_json_and_markdown(tmp_path):
    rep = verify(profile="smoke", verbose=False, out=str(tmp_path))
    files = sorted(os.listdir(tmp_path))
    assert any(f.endswith(".json") for f in files) and any(f.endswith(".md") for f in files)
    with open(tmp_path / [f for f in files if f.endswith(".json")][0]) as f:
        d = json.load(f)
    assert d["schema"] == 1 and d["profile"] == "smoke"
    assert d["counts"]["ok"] == len([r for r in rep.results if r.verdict == "ok"])
    assert d["provenance"]["root_seed"] == marks.ROOT_SEED
    assert len(d["provenance"]["marks_sha256"]) == 64
    md = rep.to_markdown()
    assert rep.summary() in md and "| check |" in md


def test_seeds_are_order_and_worker_independent():
    a = verify(profile="smoke", verbose=False, workers=1)
    b = verify(profile="smoke", verbose=False, workers=2)
    sa = {r.name: r.statistic for r in a.results}
    sb = {r.name: r.statistic for r in b.results}
    assert sa == sb
    assert seed_for("x") == seed_for("x") and seed_for("x") != seed_for("y")
    assert seed_for("x", root=1) != seed_for("x", root=2)


def test_verdict_semantics_and_exit_codes():
    with pytest.raises(ValueError):
        Result("a", "g", "MAYBE")
    with pytest.raises(ValueError):
        Result("a", "g", "SKIP")                  # a reason is mandatory
    ok = Result("a", "g", "ok", statistic=0.0)
    thin = Result("b", "g", "UNDERPOWERED", detail="2 seeds")
    bad = Result("c", "g", "FAIL", detail="x")
    approx = Result("d", "g", "EXPECTED_APPROX", detail="documented")
    assert Report("fast", [ok, approx], {}, 0.0).exit_code == 0
    assert Report("fast", [ok, thin], {}, 0.0).exit_code == 2
    assert Report("fast", [ok, thin, bad], {}, 0.0).exit_code == 1
    assert not Report("fast", [ok, thin], {}, 0.0).ok


def test_a_raising_check_is_a_failure_not_a_crash(monkeypatch):
    from winning.ratings.verify import core

    def boom(ctx):
        raise RuntimeError("kaboom")
    monkeypatch.setitem(core.CHECKS, "zz.boom",
                        core.Check("zz.boom", boom, frozenset(PROFILES), "zz", 0.1))
    rep = verify(profile="smoke", only="zz.*", verbose=False)
    assert len(rep.results) == 1 and rep.results[0].verdict == "FAIL"
    assert "RuntimeError" in rep.results[0].detail
