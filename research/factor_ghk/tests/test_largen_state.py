"""Regression tests for the two defects in the large-field workflow of PR #166.

#167: a JSON that records a Sobol truth stream as complete, paired with a
missing .npz, used to return an all-zero "truth" and never recompute it.
#169: compile_largen.py printed the first truth revision's header and the
stored errors of methods priced against an older truth.

    PYTHONPATH=. python -m pytest research/factor_ghk/tests -q
"""
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import compile_largen as CL  # noqa: E402
import hybrid_a as H  # noqa: E402
import hybrid_a_largen as L  # noqa: E402
import nodes_b as NB  # noqa: E402

N, TOTAL, CHUNK = 30, 64, 16


@pytest.fixture(scope="module")
def pool():
    mu, V, D, _ = H.make_field(N, 3, 0.05, 3)
    L.G.update(mu=mu, V=V, D=D, points=129, margin=4.0)
    with mp.get_context("fork").Pool(2) as p:
        yield p


@pytest.fixture(scope="module")
def fresh(pool, tmp_path_factory):
    d = tmp_path_factory.mktemp("fresh")
    p, used, _ = L.sobol_stream(pool, L.Arrays(str(d / "a.npz")), {}, 101, TOTAL, CHUNK)
    assert used == TOTAL and abs(p.sum() - 1) < 1e-6
    return p


def test_complete_json_without_npz_recomputes(pool, fresh, tmp_path):
    # the #167 state: JSON says done, the .npz was never committed
    state = {"sobol101": {"done": TOTAL, "seconds": 4351.0}}
    arrays = L.Arrays(str(tmp_path / "missing.npz"))
    p, used, _ = L.sobol_stream(pool, arrays, state, 101, TOTAL, CHUNK)
    assert used == TOTAL and np.count_nonzero(p) == N
    np.testing.assert_allclose(p, fresh, rtol=0, atol=1e-14)
    # and it repaired the state on disk: a second load needs no work
    again = L.Arrays(arrays.path)
    assert int(again["sobol101_done"]) == TOTAL
    np.testing.assert_allclose(again["sobol101"] / TOTAL, fresh, rtol=0, atol=1e-14)


def test_sum_without_count_is_not_double_counted(pool, fresh, tmp_path):
    arrays = L.Arrays(str(tmp_path / "a.npz"))
    arrays["sobol101"] = fresh * TOTAL                                 # a sum, but the JSON never recorded it
    p, used, _ = L.sobol_stream(pool, arrays, {}, 101, TOTAL, CHUNK)
    assert used == TOTAL
    np.testing.assert_allclose(p, fresh, rtol=0, atol=1e-14)


def test_npz_count_is_authoritative(pool, fresh, tmp_path):
    # the .npz is rewritten before the JSON; a crash between leaves the npz ahead
    arrays = L.Arrays(str(tmp_path / "a.npz"))
    arrays.update({"sobol101": fresh * TOTAL, "sobol101_done": TOTAL})
    state = {"sobol101": {"done": TOTAL // 2, "seconds": 0.0}}
    p, used, _ = L.sobol_stream(pool, arrays, state, 101, TOTAL, CHUNK)
    assert used == TOTAL and state["sobol101"]["done"] == TOTAL
    np.testing.assert_allclose(p, fresh, rtol=0, atol=1e-14)


def test_legacy_npz_is_trusted_and_stamped(pool, fresh, tmp_path):
    arrays = L.Arrays(str(tmp_path / "a.npz"))
    arrays["sobol101"] = fresh * TOTAL                                 # committed n=1e3/1e4 files have no _done
    p, used, _ = L.sobol_stream(pool, arrays, {"sobol101": {"done": TOTAL, "seconds": 1.0}}, 101, TOTAL, CHUNK)
    assert used == TOTAL
    np.testing.assert_allclose(p, fresh, rtol=0, atol=1e-14)
    assert int(L.Arrays(arrays.path)["sobol101_done"]) == TOTAL


def test_zero_sum_is_never_returned_as_truth(pool, tmp_path):
    arrays = L.Arrays(str(tmp_path / "a.npz"))
    arrays.update({"sobol101": np.zeros(N), "sobol101_done": TOTAL})
    with pytest.raises(L.StateMismatch):
        L.sobol_stream(pool, arrays, {"sobol101": {"done": TOTAL, "seconds": 0.0}}, 101, TOTAL, CHUNK)


def _state(tmp_path, n, npz):
    d = tmp_path / "runs" / "state_largen"; d.mkdir(parents=True)
    base = d / f"n{n}_rank3_D0.05_seed3_margin4.0_pts257"
    json.dump({"sobol101": {"done": 4}, "sobol202": {"done": 4}}, open(str(base) + ".json", "w"))
    if npz is not None:
        np.savez(str(base) + ".npz", **npz)


@pytest.mark.parametrize("npz", [None, {"sobol101": np.ones(5)}, {"sobol101": np.zeros(5), "sobol202": np.zeros(5)}])
def test_largen_truth_fails_clearly(tmp_path, monkeypatch, npz):
    _state(tmp_path, 5, npz)
    monkeypatch.setattr(NB, "HERE", str(tmp_path))
    with pytest.raises(NB.TruthUnavailable, match="regenerate|lack|draws"):
        NB.largen_truth(5)


def test_largen_truth_reads_a_good_state(tmp_path, monkeypatch):
    _state(tmp_path, 5, {"sobol101": np.full(5, 0.8), "sobol202": np.full(5, 0.8)})
    monkeypatch.setattr(NB, "HERE", str(tmp_path))
    p, d, key = NB.largen_truth(5)
    np.testing.assert_allclose(p, 0.2)
    assert d == 0 and key == "sobol 2^2"


def test_screen_runs_without_the_truth(tmp_path, monkeypatch):
    """The fresh-checkout path for `screen.py --n 100000`: the engine identity
    and the timings are produced, accuracy rows are skipped, the raw record
    is appended to the CSV.  Small n so it runs in a test."""
    import screen
    def missing(n):
        raise NB.TruthUnavailable("no state")
    monkeypatch.setattr(NB, "largen_truth", missing)
    out = tmp_path / "screen.csv"
    monkeypatch.setattr(sys, "argv", ["screen.py", "--n", "200", "--m", "7", "--csv", str(out)])
    screen.main()
    rows = list(csv.DictReader(open(out)))
    assert [r["row"] for r in rows] == ["vs engine", "screened Sobol"]
    assert float(rows[0]["max_diff_engine"]) < 1e-9
    assert rows[1]["all_n_max_abs"] == ""


# ------------------------------------------------------------------ #169
FIELDS = ["n", "rank", "D", "seed", "margin", "sharp", "target", "truth_p", "method", "budget", "passes", "p",
          "abs_err", "rel_err", "seconds", "truth_m", "truth_vs_scramble", "truth_vs_mc"]


def _row(method, target, truth_p, p, m, scr):
    return dict(n=100, rank=3, D=0.05, seed=3, margin=4.0, sharp=10.0, target=target, truth_p=truth_p, method=method,
                budget=32, passes=32, p=p, abs_err=abs(p - truth_p), rel_err=abs(p - truth_p) / truth_p, seconds=1.0,
                truth_m=m, truth_vs_scramble=scr, truth_vs_mc=1e-3)


def test_compile_scores_everything_against_the_latest_truth():
    old = {"1": 0.50, "2": 0.10}; new = {"1": 0.52, "2": 0.09}
    rows = [_row("only-old", t, old[t], 0.4, 15, 1e-4) for t in old]
    rows += [_row("both", t, old[t], 0.3, 15, 1e-4) for t in old]
    rows += [_row("both", t, new[t], 0.3, 17, 2e-5) for t in new]
    s = CL.compile_rows([{k: str(v) for k, v in r.items()} for r in rows])
    assert "Truth: Sobol 2^17" in s and "2.0e-05" in s and "2^15" not in s
    line = [x for x in s.splitlines() if x.startswith("| only-old † | 32")][0]
    want_abs = max(abs(0.4 - 0.52), abs(0.4 - 0.09)); want_rel = max(abs(0.4 - 0.52) / 0.52, abs(0.4 - 0.09) / 0.09)
    assert f"| {want_abs:.1e} | {want_rel:.1e} |" in line
    assert "| both | 32" in s                                             # rerun after the resume: no dagger


def test_compile_committed_n100000_uses_the_2e17_truth():
    path = HERE.parent / "runs" / "largen_n100000.csv"
    rows = list(csv.DictReader(open(path)))
    s = CL.compile_rows(rows)
    assert "Truth: Sobol 2^17" in s and "2.5e-05" in s
    r128 = [x for x in s.splitlines() if x.startswith("| Hybrid A per-winner R=128 x 12 targets † | 1536")][0]
    rel = float(r128.split("|")[5])
    assert rel == pytest.approx(0.149, abs=0.006)                     # the review's rescoring; stored rows said 0.14


def test_results_md_matches_the_compiler():
    """RESULTS.md part 2 tables are exactly what compile_largen prints."""
    text = (HERE.parent / "RESULTS.md").read_text()
    for f in sorted((HERE.parent / "runs").glob("largen_n*.csv")):
        s = CL.compile_rows(list(csv.DictReader(open(f)))).strip()
        assert s in text, f.name
