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
    expected["ok_independent"] = "a case that has always agreed"
    monkeypatch.setattr(mod, "EXPECTED", expected)
    assert mod.main() == 1
    assert "no longer diverges" in capsys.readouterr().out


def test_every_expected_entry_carries_a_reason():
    mod = _load()
    for cid, why in mod.EXPECTED.items():
        assert len(why) > 30, f"{cid} is waived without a reason"


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
