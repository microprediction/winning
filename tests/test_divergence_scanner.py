"""The scanner's own guards, sabotaged.

parity/check_divergence.py is the thing that tells us the ports agree,
so its failure modes matter more than most: every one of them ends in
the same place, a green report about a comparison that did not happen.
Each test below breaks one guard and asserts the scan goes red.
"""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(ROOT, "parity", "check_divergence.py")


def _load():
    spec = importlib.util.spec_from_file_location("check_divergence", SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_ports_or_skip(mod):
    """CI runs the ports in separate jobs, so a job may hold one
    toolchain. The scan needs two to compare anything."""
    import shutil
    have = sum(1 for cmd in mod.RUNNERS.values()
               if os.path.isabs(cmd[0]) or shutil.which(cmd[0]))
    if have < 2:
        pytest.skip(f"only {have} port toolchain available")


def _always_fails():
    """A runner that EXISTS and exits non-zero, on any machine that can
    run this suite at all -- unlike node, R or julia."""
    return [sys.executable, "-c", "raise SystemExit(3)"]


def test_a_port_that_fails_to_run_is_not_silently_dropped(monkeypatch,
                                                          capsys):
    """An ABSENT toolchain is a skip; a runner that exits non-zero is a
    failure. Collapsing the two let julia fall out of the comparison
    while the scan still printed that every case agreed."""
    mod = _load()
    _two_ports_or_skip(mod)
    runners = dict(mod.RUNNERS)
    runners["browser"] = _always_fails()
    monkeypatch.setattr(mod, "RUNNERS", runners)
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "FAILED to run the browser port" in out


def test_an_expected_entry_that_stopped_diverging_fails(monkeypatch,
                                                        capsys):
    """A waiver that outlives its defect hides the next one."""
    mod = _load()
    _two_ports_or_skip(mod)
    expected = dict(mod.EXPECTED)
    expected["ok_independent"] = {
        "why": "a case that has always agreed",
        "ports": sorted(mod.RUNNERS),
    }
    monkeypatch.setattr(mod, "EXPECTED", expected)
    assert mod.main() == 1
    out = capsys.readouterr().out
    # name the case: asserting only the exit status lets ANOTHER false
    # stale entry stand in for this one and the test still pass (#310)
    assert "'ok_independent' no longer diverges" in out


def test_a_waiver_is_not_stale_when_its_port_is_absent(monkeypatch,
                                                       capsys):
    """A waiver exists because some port differs. With that port absent,
    the survivors agreeing says nothing -- and telling the reader to
    delete a live waiver is worse than not checking (#310)."""
    mod = _load()
    runners = {k: v for k, v in mod.RUNNERS.items() if k != "julia"}
    if len(runners) < 2:
        pytest.skip("need two other port toolchains")
    monkeypatch.setattr(mod, "RUNNERS", runners)
    waived = [c for c, e in mod.EXPECTED.items() if "julia" in e["ports"]]
    if not waived:
        pytest.skip("no waiver currently depends on julia")
    rc = mod.main()
    out = capsys.readouterr().out
    assert "no longer diverges" not in out
    assert "not judged" in out and "julia" in out
    assert rc == 0, out


def test_a_waiver_for_a_deleted_case_is_reported(monkeypatch, capsys):
    """A waiver is only examined while walking the cases, so one whose
    case has been removed is never looked at again. Found reviewing my
    own diff, not by it failing: the stale check runs one step too
    late to catch it."""
    mod = _load()
    _two_ports_or_skip(mod)
    expected = dict(mod.EXPECTED)
    expected["a_case_that_was_deleted"] = {
        "why": "a waiver whose case no longer exists in the file",
        "ports": sorted(mod.RUNNERS),
    }
    monkeypatch.setattr(mod, "EXPECTED", expected)
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "'a_case_that_was_deleted' is not in the case file" in out


def test_every_waiver_names_the_ports_it_is_about():
    mod = _load()
    for cid, entry in mod.EXPECTED.items():
        assert set(entry) >= {"why", "ports"}, cid
        assert entry["ports"], f"{cid} names no ports"
        assert set(entry["ports"]) <= set(mod.RUNNERS), cid


def test_every_expected_entry_carries_a_reason():
    mod = _load()
    for cid, entry in mod.EXPECTED.items():
        assert len(entry["why"]) > 30, f"{cid} is waived without a reason"


def test_values_are_compared_not_only_verdicts():
    """Two ports can agree to ACCEPT and answer differently; that is the
    quieter half of the same defect."""
    mod = _load()
    assert mod._value_gap({"a": [1.0, 2.0], "b": [1.0, 2.0]}) == 0.0
    assert mod._value_gap({"a": [1.0], "b": [1.5]}) == pytest.approx(0.5)
    # a different SHAPE is a disagreement, not something to zip past
    assert mod._value_gap({"a": [1.0], "b": [1.0, 2.0]}) == float("inf")
    # one port alone cannot disagree with anybody
    assert mod._value_gap({"a": [1.0]}) is None


def test_the_inverse_is_not_compared_tighter_than_it_converges():
    """Both inverses stop at a 1e-8 residual, so demanding 1e-9 of their
    outputs tests the arithmetic rather than the port."""
    mod = _load()
    assert mod.VERB_TOL["inverse"] > mod.VALUE_TOL
    assert mod.VERB_TOL["inverse"] <= 1e-6


def test_the_case_file_covers_every_verb_the_runners_implement():
    cases = json.load(open(os.path.join(ROOT, "parity",
                                        "divergence_cases.json")))["cases"]
    verbs = {c["verb"] for c in cases}
    assert {"race", "inverse", "topk", "rank", "bottomk", "hermite"} <= verbs
    assert len(cases) >= 44
