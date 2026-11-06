"""The classic inverse keeps its 1.6.0 positional API, and the wrappers
validate nIter and tol on every book (#924).

#904 inserted ``tol`` between ``nIter`` and ``verbose``, so the 1.6.0
call ``solve_for_implied_offsets(p, d, None, None, 3, True)`` read
``True`` as the tolerance and raised. ``tol`` is now keyword-only after
``visualize``. And a book with one live runner returned before the
inverse ran, so ``nIter=0`` and ``tol=0`` were accepted there.
"""
import inspect

import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density
from winning.classic.lattice_calibration import (CALIBRATION_MAX_ITER, CALIBRATION_TOL,
                                                 dividend_implied_ability, solve_for_implied_offsets,
                                                 state_price_implied_ability)

D = skew_normal_density(20, 0.1)


def test_positional_parameters_are_the_1_6_0_ones():
    params = inspect.signature(solve_for_implied_offsets).parameters
    positional = [n for n, p in params.items() if p.kind == p.POSITIONAL_OR_KEYWORD]
    assert positional == ['prices', 'density', 'offset_samples', 'implied_offsets_guess',
                          'nIter', 'verbose', 'visualize']
    assert params['tol'].kind == params['tol'].KEYWORD_ONLY
    assert params['nIter'].default == CALIBRATION_MAX_ITER
    assert params['tol'].default == CALIBRATION_TOL


def test_the_1_6_0_positional_verbose_call_runs(capsys):
    mu = solve_for_implied_offsets([0.6, 0.3, 0.1], D, None, None, 3, True)
    assert capsys.readouterr().out.strip()          # verbose printed its diagnostics
    np.testing.assert_array_equal(mu, solve_for_implied_offsets([0.6, 0.3, 0.1], D, nIter=3))


def test_tol_is_still_taken_by_keyword():
    loose = solve_for_implied_offsets([0.6, 0.3, 0.1], D, tol=1e-3)
    tight = solve_for_implied_offsets([0.6, 0.3, 0.1], D)
    assert not np.array_equal(loose, tight)
    with pytest.raises(TypeError):
        solve_for_implied_offsets([0.6, 0.3, 0.1], D, None, None, 3, False, False, 1e-3)


@pytest.mark.parametrize("kw", [{"nIter": 0}, {"nIter": -1}, {"nIter": 2.5}, {"nIter": True},
                                {"tol": 0}, {"tol": -1e-6}, {"tol": float("nan")}, {"tol": None}])
def test_one_live_runner_still_validates_the_controls(kw):
    name = next(iter(kw))
    with pytest.raises((ValueError, TypeError), match=name):
        dividend_implied_ability([2.0], D, **kw)
    with pytest.raises((ValueError, TypeError), match=name):
        state_price_implied_ability([1.0, 0.0], D, **kw)


def test_one_live_runner_with_valid_controls_is_unchanged():
    assert dividend_implied_ability([2.0], D) == [0.0]
    assert dividend_implied_ability([2.0], D, nIter=5, tol=1e-6) == [0.0]
    assert state_price_implied_ability([1.0, 0.0], D) == [0.0, float('inf')]
