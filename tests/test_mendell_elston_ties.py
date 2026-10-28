"""Mendell-Elston is label free on exact ties (#287), and its jitter is
relative to the problem's scale (#375)."""
import numpy as np
import pytest

from winning.methods.orthant_extra import _diff_problem, _mendell_elston


def _probs(mu, S):
    p = np.array([_mendell_elston(*_diff_problem(mu, S, i)) for i in range(len(mu))])
    return p / p.sum()


B4 = np.array([[1, -2], [1, 0], [2, 1], [2, 2]], float)
B5 = np.array([[-0.1975505311, 0.2465199213, 0.1450564050],
               [-0.4251065014, -0.8689925848, -0.1397214371],
               [0.9685102219, -0.2744182122, 0.6584296166],
               [1.2751405253, 2.2332542156, 0.1289912994],
               [-2.5677020449, 0.7720179885, -0.1099585960]])
D5 = np.array([0.42060100, 0.15034860, 0.21494677, 0.19269718, 0.16847565])


@pytest.mark.parametrize("C,perm", [
    (B4 @ B4.T + 0.5 * np.eye(4), [3, 1, 0, 2]),
    (B4 @ B4.T + 0.5 * np.eye(4), [1, 0, 3, 2]),
    (B5 @ B5.T + np.diag(D5), [2, 4, 1, 0, 3]),
])
def test_equal_abilities_are_label_free(C, perm):
    mu = np.zeros(len(C))
    p = _probs(mu, C)
    q = _probs(mu[perm], C[np.ix_(perm, perm)])
    back = np.empty_like(q)
    back[perm] = q
    assert np.abs(p - back).max() < 1e-14


def test_the_jitter_is_scale_free():
    # max-wins: the same race at d and at 1e-14 * d
    for d in (1.0, 1e-14):
        s = np.sqrt(d)
        mu = s * np.array([1.0, 0.0, -1.0])
        C = np.ones((3, 3)) + d * np.eye(3)
        p = _probs(mu, C)
        if d == 1.0:
            ref = p
    assert np.abs(p - ref).max() < 1e-2
