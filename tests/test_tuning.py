"""Selecting a hyperparameter, and saying whether the sweep did anything."""
import pytest

from winning.ratings import TuningReport, require_live, select, select_grid


def test_inert_parameter_is_flagged_and_raises():
    """Glicko-2's tau on one month of Lichess: every value ties."""
    tau = {0.02: 0.6273, 0.05: 0.6273, 0.1: 0.6273, 0.2: 0.6273,
           0.3: 0.6273, 0.5: 0.6273, 1.0: 0.6273}
    best, report = select(tau, "tau")
    assert report.inert
    assert not report.trustworthy
    assert "INERT" in str(report)
    assert best == 0.02          # min takes the first of a tie
    with pytest.raises(AssertionError, match="tie-break"):
        require_live(report)


def test_saturated_edge_is_benign():
    """A recency parameter past the point where it switches decay off."""
    period = {200.0: 0.6203, 1500.0: 0.6200, 20000.0: 0.6199}
    best, report = select(period, "period")
    assert best == 20000.0
    assert report.at_edge and report.saturated and not report.inert
    assert report.trustworthy
    assert "SATURATED" in str(report)
    require_live(report)


def test_unresolved_edge_asks_for_a_wider_grid():
    ird = {350.0: 0.6273, 500.0: 0.6239, 700.0: 0.6216}
    best, report = select(ird, "initial_rd")
    assert best == 700.0
    assert report.at_edge and not report.saturated
    assert not report.trustworthy
    assert "GRID EDGE" in str(report)


def test_interior_selection_is_ok():
    offset = {3.0: 0.6085, 10.0: 0.6076, 30.0: 0.6076,
              100.0: 0.6078, 300.0: 0.6080}
    best, report = select(offset, "offset ridge")
    assert best == 10.0
    assert report.trustworthy
    assert not report.at_edge and not report.inert


def test_order_decides_the_ends_not_insertion():
    scores = {1.0: 0.50, 0.1: 0.40, 10.0: 0.60}
    _, unordered = select(scores, "lam")
    assert not unordered.at_edge          # 0.1 sits in the middle as inserted
    _, ordered = select(scores, "lam", order=[0.1, 1.0, 10.0])
    assert ordered.at_edge                # and at the end once sorted


def test_order_must_cover_the_scores():
    with pytest.raises(KeyError):
        select({1.0: 0.5, 2.0: 0.4}, "lam", order=[1.0, 3.0])


def test_empty_is_an_error():
    with pytest.raises(ValueError):
        select({}, "lam")
    with pytest.raises(ValueError):
        select_grid({}, ["a"])


def test_single_value_is_not_tuning():
    best, report = select({0.5: 0.61}, "lam")
    assert best == 0.5 and report.inert and not report.trustworthy


def test_grid_reports_each_axis_separately():
    """A tuple at an interior position can still hide an axis at its end."""
    # Strictly decreasing in period so the axis has a real optimum at
    # its top end; interior optimum in initial_rd at 500. An earlier
    # version of this fixture tied two period values, and min silently
    # took the first: the exact failure this module exists to catch.
    per_effect = {30.0: 0.004, 200.0: 0.001, 1500.0: 0.000}
    ird_effect = {350.0: 0.0004, 500.0: 0.0000, 700.0: 0.0002}
    scores = {(p, i): 0.63 + per_effect[p] + ird_effect[i]
              for p in per_effect for i in ird_effect}
    best, reports = select_grid(scores, ["period", "initial_rd"])
    assert set(reports) == {"period", "initial_rd"}
    assert best[0] == 1500.0 and best[1] == 500.0
    assert reports["period"].at_edge          # top of its own axis
    assert not reports["initial_rd"].at_edge  # interior on its own axis


def test_grid_name_count_must_match():
    with pytest.raises(ValueError):
        select_grid({(1.0, 2.0): 0.5}, ["only_one"])


def test_report_is_a_named_tuple_so_it_can_be_logged():
    _, report = select({1.0: 0.5, 2.0: 0.4, 3.0: 0.45}, "lam")
    assert isinstance(report, TuningReport)
    assert report.name == "lam" and report.best == 2.0
    assert report.spread == pytest.approx(0.1)
