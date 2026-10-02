"""Regression for research/contest_design/exp2_investment: investment
lands after the current round, so in a one-round contest it can never
pay and nobody invests (the first version valued a phantom final round
and invested 0.0046 per tercile at T = 1)."""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "research", "contest_design")


def _load():
    sys.path.insert(0, os.path.join(ROOT, "exp1_prize_schedules"))
    spec = importlib.util.spec_from_file_location(
        "contest_design_run_invest", os.path.join(ROOT, "exp2_investment", "run_invest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_round_contest_has_no_investment():
    ri = _load()
    field = np.sort(np.random.default_rng(0).normal(0, ri.SIGMA_A, ri.N))
    ri.T = 1
    r = ri.simulate(field, ri.schedule("wta"), lam=0.0, lag=0, seed=0)
    assert r["ability_gain_by_tercile"] == [0.0, 0.0, 0.0]


def test_last_round_has_no_investment_but_earlier_rounds_do():
    ri = _load()
    field = np.sort(np.random.default_rng(0).normal(0, ri.SIGMA_A, ri.N))
    ri.T = 2
    r = ri.simulate(field, ri.schedule("wta"), lam=0.0, lag=0, seed=0)
    assert sum(r["ability_gain_by_tercile"]) > 0
