"""winning.ratings.simulate: the samplers match the densities that define
the bases, the max-wins flip is the right way round, and the worlds
reproduce the private helpers they replaced (those tests' pinned numbers
are the proof; here the bandits audit world is pinned draw for draw)."""
import numpy as np
import pytest

from winning.factor.races import (exponential_power_base, failure_base,
                                  skew_logistic_base, skew_normal_base,
                                  student_base)
from winning.ratings.simulate import (check_sampler, independent_world,
                                      sample_min, sample_noise)

BASES = [("normal", "normal"), ("gumbel", "gumbel"), ("logistic", "logistic"),
         ("laplace", "laplace"), ("student4", student_base(4.0)),
         ("failure", failure_base(0.15)),
         ("failure_std", failure_base(0.15, standardize=True)),
         ("failure_on_laplace", failure_base(0.2, base="laplace")),
         ("expo1.5", exponential_power_base(1.5)),
         ("expo6", exponential_power_base(6.0)),
         ("skewlog0.5", skew_logistic_base(0.5)),
         ("skewlog3", skew_logistic_base(3.0)),
         ("skewnorm2", skew_normal_base(2.0))]


@pytest.mark.parametrize("name,base", BASES, ids=[b[0] for b in BASES])
def test_sampler_matches_its_own_survival_function(name, base):
    # DKW: an exact sampler's Kolmogorov distance at n = 50,000 is below
    # 7.5e-3 with probability 0.999; a sign or scale slip is > 0.1
    rng = np.random.default_rng(hash(name) % 2**32)
    assert check_sampler(base, rng, n=50_000) < 7.5e-3


def test_named_bases_are_standardized_and_gumbel_flips_the_right_way():
    rng = np.random.default_rng(1)
    for name in ("normal", "gumbel", "logistic", "laplace"):
        z = sample_min(rng, name, 200_000)
        assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.02
    # min-wins Gumbel base has its heavy tail on the LEFT (S(z) =
    # exp(-e^{cz - gamma}) dies doubly-exponentially on the right), so
    # the max-wins draw is right-skewed: eps = -Z has positive skew
    e = sample_noise(rng, "gumbel", 200_000)
    assert ((e - e.mean()) ** 3).mean() > 0.5


def test_callable_without_sampler_is_refused():
    def bare(z):
        return np.ones_like(z), np.ones_like(z), np.zeros_like(z)
    with pytest.raises(ValueError):
        sample_min(np.random.default_rng(0), bare, 5)


def test_independent_world_reproduces_the_bandits_audit_draws():
    # bandits/tests/audit_ratings_bulletproof.py::_world/_events, inline
    M, K, n = 8, 5, 6
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, M); a = a - a.mean()
    ref = []
    for _ in range(n):
        S = rng.choice(M, K, replace=False)
        perf = a[S] + rng.normal(0, 1, K)
        ref.append((S, np.argsort(-perf)))
    a2, evs = independent_world(np.random.default_rng(3), M, K, n,
                                noise="gaussian")
    assert np.abs(a2 - a).max() == 0.0
    for (S, o), (S2, o2) in zip(ref, evs):
        assert (S == S2).all() and (o == o2).all()


def test_independent_world_regimes_scale_as_declared():
    rng = np.random.default_rng(4)
    a, evs = independent_world(rng, M=200, K=5, n=3, prior_var=9.0)
    assert 2.0 < a.std() < 4.0
    V = np.full((6, 1), 0.6)
    a, evs = independent_world(np.random.default_rng(5), M=6, K=6, n=4, V=V,
                               base="laplace")
    assert len(evs) == 4 and all(len(o) == 6 for _, o in evs)
