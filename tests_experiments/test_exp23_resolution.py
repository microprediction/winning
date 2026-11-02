"""Research timings must compare the same explicit lattice resolution (#960)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'research/experiments/exp23_newton_cg'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('points', [257, 501])
def test_per_call_shared_resolution(monkeypatch, points):
    module = load('run_field_size_scaling')
    seen = []
    def record(*args, **kwargs):
        seen.append(kwargs['points'])
    monkeypatch.setattr(module, 'race_probabilities', record)
    monkeypatch.setattr(module, 'jacobian_vector_product', record)
    module.part1_per_call_cost(points=points, cases=((3, 1),))
    assert seen == [points, points]


@pytest.mark.parametrize('points', [257, 501])
def test_hybrid_threads_points_to_every_call(monkeypatch, points):
    module = load('run_newton_cg3')
    seen = []
    def forward(mu, **kwargs):
        seen.append(('forward', kwargs['points']))
        return np.array([.4, .6]), np.array([-.2, -.2])
    def jvp(mu, V, D, F, W, h, **kwargs):
        seen.append(('jvp', kwargs['points']))
        return -np.asarray(h)
    monkeypatch.setattr(module, 'race_probabilities', forward)
    monkeypatch.setattr(module, 'jacobian_vector_product', jvp)
    module.hybrid(np.array([.3, .7]), None, None, None, None,
                  warm=1, max_newton=1, max_cg=1, points=points)
    assert sum(kind == 'forward' for kind, _ in seen) == 5
    assert ('jvp', points) in seen
    assert all(value == points for _, value in seen)


@pytest.mark.parametrize('points', [257, 501])
def test_trial_target_and_both_solvers_share_points(monkeypatch, points):
    module = load('run_newton_cg3')
    seen = []
    def target(*args, **kwargs):
        seen.append(('target', kwargs['points']))
        return np.array([.3, .7])
    def jacobi(*args, **kwargs):
        seen.append(('jacobi', kwargs['points']))
        return None, dict(iterations=1, max_log_residual=0.)
    def hybrid(*args, **kwargs):
        seen.append(('hybrid', kwargs['points']))
        return None, dict(newton=1, jvp=1, residual=0., converged=True)
    monkeypatch.setattr(module, 'race_probabilities', target)
    monkeypatch.setattr(module, 'abilities_from_race', jacobi)
    monkeypatch.setattr(module, 'hybrid', hybrid)
    module.trial('test', np.zeros(2), None, None, None, None, points=points)
    assert seen == [('target', points), ('jacobi', points), ('hybrid', points)]
