"""Tie densities are not normalised, so the grid must resolve the
narrowest runner whatever the span: an irrelevant distant runner cut a
close pair's density by 10.3% at 501 points (#349)."""
import numpy as np
from scipy.stats import norm

from winning.factor.races import tie_densities


def test_a_distant_runner_does_not_dilute_a_close_pair():
    F, W = np.array([[0.0]]), np.array([1.0])
    mu2, D2 = np.array([0.0, 0.1]), np.array([0.01, 0.01])
    exact = norm.pdf(mu2[0] - mu2[1], scale=np.sqrt(D2.sum()))
    w3 = tie_densities(np.array([0.0, 0.1, 100.0]), V=np.zeros((3, 1)),
                       D=np.array([0.01, 0.01, 1.0]), F=F, W=W, points=501)
    assert abs(w3[0, 1] - exact) < 1e-9
