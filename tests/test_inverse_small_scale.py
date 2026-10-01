"""abilities_from_race on small-scale fields, and the exact pair inverse.

Handed over by a client 2026-09-22: two names with near-identical high
loadings, b = 0.99, D = 1 - b^2 -- the race is a single contrast with sd
0.199 -- came back with gap 1.2755 for an exact 0.1679, forward map
[1.0, 0.0], after 60 non-converging sweeps. b = 0.80 and 0.95 were exact.
Measured: the same divergence at n = 2 and n = 3 with D = 0.005 / 0.001
and no loadings at all, while n >= 4 held. So it was SCALE, not the pair:
the warm start and the step cap were in unit-variance units.

Both now scale with the field's contrast sd, and the normal pair takes
the closed form (the mirror of the forward one): mu1 - mu0 = sd_d Phi^-1(p0).
Fields near unit scale are untouched -- the sweep counts pinned in
test_inverse_effective_pair.py did not move.
"""
import warnings

import numpy as np
import pytest
from scipy.special import ndtri

import winning.factor as wf
from winning.factor.races import skew_normal_base


@pytest.mark.parametrize("b", [0.80, 0.95, 0.99, 0.999])
def test_two_names_with_high_common_loading_are_exact(b):
    V = np.array([[b], [b]]); D = np.full(2, 1 - b * b)
    mu, info = wf.abilities_from_race(np.array([0.8, 0.2]), V=V, D=D,
                                      target_floor=1e-12, return_info=True)
    exact_gap = np.sqrt(2 * (1 - b * b)) * ndtri(0.8)
    assert info["converged"] and info["iterations"] == 0        # closed form
    assert abs((mu[1] - mu[0]) - exact_gap) < 1e-12
    assert abs(mu.sum()) < 1e-12                                 # mean-zero gauge
    q = np.asarray(wf.race_probabilities(mu, V=V, D=D))
    assert np.abs(q - [0.8, 0.2]).max() < 1e-9


def test_pair_closed_form_matches_the_forward_on_a_rank_two_pair():
    V = np.array([[0.7, -0.2], [-0.3, 0.5]]); D = np.array([0.6, 0.9])
    for p0 in (0.5, 0.7, 0.9, 0.999):
        p = np.array([p0, 1 - p0])
        mu = wf.abilities_from_race(p, V=V, D=D)
        q = np.asarray(wf.race_probabilities(mu, V=V, D=D, points=257))
        assert np.abs(q - p).max() < 1e-9


@pytest.mark.parametrize("n,D_level", [(2, 0.005), (2, 0.001), (3, 0.005), (3, 0.001), (4, 0.001)])
def test_small_scale_fields_converge(n, D_level):
    """n=2 and n=3 at these scales diverged (residual 0.2-0.5)."""
    rng = np.random.default_rng(0)
    p = np.sort(rng.dirichlet(np.ones(n) * 2))[::-1]
    D = np.full(n, D_level)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # no non-convergence warning
        mu, info = wf.abilities_from_race(p, D=D, return_info=True)
    q = np.asarray(wf.race_probabilities(mu, D=D))
    assert info["converged"]
    assert np.abs(q - p).max() < 1e-6


def test_the_scaled_loop_serves_a_small_scale_pair_under_a_non_normal_base():
    """The closed form is normal-only; a skew pair at sd ~0.1 goes through
    the loop, which must now converge too."""
    D = np.full(2, 0.005); base = skew_normal_base(2.0)
    p = np.array([0.75, 0.25])
    mu, info = wf.abilities_from_race(p, D=D, base=base, return_info=True)
    q = np.asarray(wf.race_probabilities(mu, D=D, base=base))
    assert info["converged"]
    assert np.abs(q - p).max() < 1e-6


def test_unit_scale_fields_are_untouched():
    """scale == 1 exactly for D = 1 with no loadings, so the start and the
    cap are bit-identical to before: same sweeps, same answer."""
    rng = np.random.default_rng(1)
    p = np.exp(rng.normal(size=8) * 0.8); p /= p.sum()
    mu, info = wf.abilities_from_race(p, D=np.ones(8), return_info=True)
    q = np.asarray(wf.race_probabilities(mu, D=np.ones(8)))
    assert info["converged"] and np.abs(q - p).max() < 1e-8
