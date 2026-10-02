"""#123: solve_for_implied_offsets(verbose=True) selects the Python
iteration, which crashed on py3 (`zip(...)[:5]`). It must run, print its
diagnostics, and agree with the default (compiled when present) path."""
import numpy as np

import winning
from winning.classic.lattice import skew_normal_density
from winning.classic.lattice_calibration import solve_for_implied_offsets


def test_verbose_calibration_runs_and_matches_default(capsys):
    d = skew_normal_density(L=100, unit=0.05, a=0)
    prices = [.5, .3, .2]
    quiet = solve_for_implied_offsets(prices, d, nIter=2)
    loud = solve_for_implied_offsets(prices, d, nIter=2, verbose=True)
    out = capsys.readouterr().out
    assert "(" in out                       # the (price, guess) pairs
    assert np.allclose(np.asarray(loud, float), np.asarray(quiet, float),
                       atol=1e-9)


def test_verbose_calibration_pure_backend(capsys):
    d = skew_normal_density(L=100, unit=0.05, a=0)
    try:
        winning.use_rust(False)
        solve_for_implied_offsets([.5, .3, .2], d, nIter=1, verbose=True)
    finally:
        winning.use_rust(True)
    assert capsys.readouterr().out
