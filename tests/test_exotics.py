"""Anchors and identities for ordered_probabilities / harville_prefix_logprob.
Exact identities wherever possible; Monte Carlo nowhere."""
import numpy as np
import pytest

from winning.factor.exotics import ordered_probabilities, harville_prefix_logprob
from winning.factor.races import race_probabilities, softmax_probabilities
from winning.factor.topk import top_k_probabilities

GUMBEL_D = np.pi ** 2 / 6.0


def test_section9_reverse_order_anchor():
    # methodology note, Sec 9: means (-2,0,2), unit variance, reverse order
    # (index 2 first, 1 second, 0 third). Published q(0) = 0.0003113102393.
    q = ordered_probabilities(np.array([-2.0, 0.0, 2.0]), k=3, points=4001)
    assert abs(q[2, 1, 0] - 0.0003113102393) / 0.0003113102393 < 2e-4


def test_section9_inflation_table():
    mu = np.array([-2.0, 0.0, 2.0])
    q0 = ordered_probabilities(mu, k=3, points=4001)[2, 1, 0]
    for t, shade in ((0.01, 0.9539), (0.10, 0.6464), (1.00, 0.0823)):
        qt = ordered_probabilities(mu, k=3, D=np.full(3, 1.0 + t), points=4001)[2, 1, 0]
        assert abs(q0 / qt - shade) < 2e-3


@pytest.mark.parametrize("k", [1, 2, 3])
def test_total_mass_is_one(k):
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1, 8); D = np.exp(rng.uniform(-0.5, 0.5, 8)) ** 2
    q = ordered_probabilities(mu, k=k, D=D)
    assert abs(q.sum() - 1.0) < 1e-6


def test_k1_is_the_win_race():
    rng = np.random.default_rng(1)
    mu = rng.normal(0, 1, 7); D = np.exp(rng.uniform(-0.5, 0.5, 7)) ** 2
    p = race_probabilities(mu, D=D, points=1001)
    q = ordered_probabilities(mu, k=1, D=D, points=1001)
    assert np.max(np.abs(p - q)) < 2e-4


def test_marginal_top3_matches_topk():
    rng = np.random.default_rng(2)
    mu = rng.normal(0, 1, 7)
    q = ordered_probabilities(mu, k=3, points=1501)
    top3 = q.sum(axis=(1, 2)) + q.sum(axis=(0, 2)) + q.sum(axis=(0, 1))
    ref = top_k_probabilities(mu, 3, points=1025)
    assert np.max(np.abs(top3 - ref)) < 2e-3


def test_gumbel_base_is_harville():
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 1, 6); tau = 0.7
    q = ordered_probabilities(mu, k=3, D=np.full(6, GUMBEL_D * tau * tau),
                              base="gumbel", points=2001)
    for i in range(6):
        for j in range(6):
            for l in range(6):
                if len({i, j, l}) < 3:
                    continue
                h = np.exp(harville_prefix_logprob(mu, [i, j, l], temperature=tau))
                assert abs(q[i, j, l] - h) < 3e-3 * max(h, 1e-3)


def test_prefix_k1_is_softmax():
    mu = np.array([0.3, -0.5, 0.9, 0.0])
    p = softmax_probabilities(mu, temperature=0.8)
    for i in range(4):
        assert abs(np.exp(harville_prefix_logprob(mu, [i], temperature=0.8)) - p[i]) < 1e-12


def test_tempered_k1_matches_race_probabilities():
    rng = np.random.default_rng(4)
    mu = rng.normal(0, 1, 6)
    p = race_probabilities(mu, temperature=0.4, points=2001)
    q = ordered_probabilities(mu, k=1, temperature=0.4, points=2001)
    assert np.max(np.abs(p - q)) < 2e-3


def test_rust_matches_numpy_path():
    import winning.factor.races as R
    from winning.factor import exotics as E
    if not (R._HAVE_RUST and hasattr(R._fastrace, "ordered_prefixes")):
        pytest.skip("fastrace without ordered_prefixes")
    rng = np.random.default_rng(5)
    mu = rng.normal(0, 1, 9); D = np.exp(rng.uniform(-0.5, 0.5, 9)) ** 2
    q_rs = ordered_probabilities(mu, k=3, D=D, points=1001)
    saved = E._HAVE_RUST
    try:
        E._HAVE_RUST = False
        q_np = ordered_probabilities(mu, k=3, D=D, points=1001)
    finally:
        E._HAVE_RUST = saved
    assert np.max(np.abs(q_rs - q_np)) < 1e-9
