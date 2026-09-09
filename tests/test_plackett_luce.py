"""Harville / Plackett-Luce: the ordering law of the Gumbel race.

The stagewise formulas here are exact ONLY for Gumbel noise (Peter:
"we need Gumbel noise to replicate Harville"), and the tests enforce
that reading: the order likelihood is checked against brute-force
enumeration and against frequency simulation of an actual Gumbel race;
the place formulas against enumeration of the PL law.
"""
import itertools

import numpy as np

from winning.factor.races import (harville_order_logprob,
                                  harville_place_probabilities,
                                  softmax_probabilities)


def test_order_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=4)
    total = sum(np.exp(harville_order_logprob(mu, perm))
                for perm in itertools.permutations(range(4)))
    assert abs(total - 1.0) < 1e-12


def test_matches_gumbel_race_frequencies():
    # simulate the actual min-Gumbel race; ordering frequencies must
    # follow Harville (this is the IIA property, and it fails for any
    # other base -- which is the point of keeping these formulas
    # gumbel-only)
    rng = np.random.default_rng(1)
    n, M = 4, 400_000
    mu = np.array([-0.5, 0.0, 0.3, 0.8])
    G = np.log(-np.log1p(-rng.random((M, n))))
    order_idx = np.argsort(mu + G, axis=1)
    from collections import Counter
    counts = Counter(map(tuple, order_idx))
    worst = 0.0
    for perm, c in counts.items():
        p_hat = c / M
        p = np.exp(harville_order_logprob(mu, list(perm)))
        worst = max(worst, abs(p_hat - p))
    assert worst < 3e-3


def test_place_probabilities_match_enumeration():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=5)
    p = softmax_probabilities(mu)
    for k in (1, 2, 3):
        target = np.zeros(5)
        for perm in itertools.permutations(range(5)):
            pr = np.exp(harville_order_logprob(mu, perm))
            for pos in range(k):
                target[perm[pos]] += pr
        got = harville_place_probabilities(p, k=k)
        assert np.abs(got - target).max() < 1e-12
        assert abs(got.sum() - k) < 1e-9


def test_mixed_pl_matches_conditional_average():
    # with factor loadings, the order likelihood is the mixture of
    # conditional Harville probabilities over the supplied nodes
    rng = np.random.default_rng(3)
    n = 5
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 0.5
    from winning.factor.core import hermite_nodes
    F, W = hermite_nodes(2, 15)
    order = [2, 0, 4, 1, 3]
    lp = harville_order_logprob(mu, order, V=V, F=F, W=W)
    ref = 0.0
    for q in range(len(F)):
        ref += W[q] * np.exp(harville_order_logprob(mu + F[q] @ V.T, order))
    assert abs(np.exp(lp) - ref) < 1e-14


def test_place_sums_exact_under_near_certain_favorite():
    # gap-stress catch: 1 - p_fav computed by subtraction lost three
    # digits at p_fav = 1 - 1e-13 and the identity sum(top-k) = k
    # drifted to k + 3e-3; complements are now sums of the others
    p = np.array([1 - 1e-13, 3e-14, 3e-14, 4e-14])
    for k in (2, 3):
        out = harville_place_probabilities(p, k=k)
        assert abs(out.sum() - k) < 1e-9
        assert (out <= 1 + 1e-12).all()
    # and the deeper variant: two large entries exhausting the field
    # (denom2 cancellation), plus a 300-field fuzz of extreme spreads
    p2 = np.array([0.7, 0.3 - 4e-15, 1e-15, 1e-15, 1e-15, 1e-15])
    p2 = p2 / p2.sum()
    assert abs(harville_place_probabilities(p2, 3).sum() - 3) < 1e-9
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(3, 12))
        logp = rng.normal(size=n) * rng.uniform(1, 15)
        q = np.exp(logp - logp.max()); q /= q.sum()
        for k in (2, 3):
            assert abs(harville_place_probabilities(q, k).sum() - k) < 1e-8
