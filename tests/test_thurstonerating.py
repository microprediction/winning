"""Behavioural tests specific to the lattice Thurstone rater."""

from winning import ThurstoneRating, gaussian_win_probabilities


def test_two_player_probability_moves_sensibly():
    tr = ThurstoneRating()
    for _ in range(10):
        tr.observe(["a", "b"], [1, 2])
    p = tr.win_probabilities(["a", "b"])
    assert 0.6 < p[0] < 0.95


def test_uncertainty_decreases_with_evidence():
    tr = ThurstoneRating()
    s0 = tr.rating("a").sigma
    for _ in range(8):
        tr.observe(["a", "b", "c", "d"], [1, 2, 3, 4])
    assert tr.rating("a").sigma < s0


def test_field_strength_matters():
    """Beating strong opponents should raise a rating more than beating weak ones."""
    strong = ThurstoneRating()
    weak = ThurstoneRating()
    # s1, s2 keep beating mugs; w1, w2 keep losing to mugs
    for _ in range(12):
        strong.observe(["s1", "s2", "mug1", "mug2"], [1, 2, 3, 4])
        weak.observe(["mug3", "mug4", "w1", "w2"], [1, 2, 3, 4])
    # a new player then beats each pair
    for _ in range(3):
        strong.observe(["newbie", "s1", "s2"], [1, 2, 3])
        weak.observe(["newbie", "w1", "w2"], [1, 2, 3])
    assert strong.rating("newbie").mu > weak.rating("newbie").mu


def test_drift_reinflates_uncertainty():
    tr = ThurstoneRating(tau=0.3)
    for _ in range(8):
        tr.observe(["a", "b"], [1, 2], dt=1.0)
    s_tight = tr.rating("a").sigma
    tr.observe(["c", "d"], [1, 2], dt=50.0)  # long gap, a not involved
    s_later = tr.rating("a").sigma
    assert s_later > s_tight


def test_transitivity_of_inferred_ratings():
    tr = ThurstoneRating()
    for _ in range(6):
        tr.observe(["a", "b"], [1, 2])
        tr.observe(["b", "c"], [1, 2])
    p = tr.win_probabilities(["a", "c"])
    assert p[0] > 0.6  # a should be favored over c without ever meeting


def test_tie_updates_are_permutation_invariant():
    a = ThurstoneRating()
    b = ThurstoneRating()
    for _ in range(6):
        a.observe(["x", "y", "z"], [1, 1, 3])
        b.observe(["y", "x", "z"], [1, 1, 3])  # tied pair listed in reverse
    assert abs(a.rating("x").mu - b.rating("x").mu) < 1e-12
    assert abs(a.rating("x").mu - a.rating("y").mu) < 1e-12  # tied peers move together
    assert a.rating("x").mu > a.rating("z").mu


def test_queries_are_read_only():
    tr = ThurstoneRating()
    tr.observe(["a", "b"], [1, 2])
    tr.win_probabilities(["a", "stranger"])
    tr.rating("typo")
    assert set(tr.known()) == {"a", "b"}
    assert [nm for nm, _ in tr.leaderboard()] == ["a", "b"]


def test_elapse_before_prediction_diffuses():
    tr = ThurstoneRating(tau=0.3)
    for _ in range(8):
        tr.observe(["a", "b"], [1, 2], dt=1.0)
    p_now = tr.win_probabilities(["a", "b"])[0]
    tr.elapse(200.0)  # long idle gap widens beliefs before the next prediction
    p_later = tr.win_probabilities(["a", "b"])[0]
    assert 0.5 < p_later < p_now


def test_asymmetric_base_kernel_matches_exact_bayes():
    """The stage likelihood is a correlation with the base density; an
    asymmetric (skewed) kernel must not be silently mirrored."""
    import numpy as np

    unit, half = 0.1, 60
    x = np.arange(-half, half + 1) * unit
    kern = np.exp(-0.5 * (x / 0.8) ** 2) * (x <= 0.5)  # hard-truncated: asymmetric
    kern = kern / kern.sum()

    tr = ThurstoneRating(tau=0.0, iterations=1, base_kernel=kern)
    tr.observe(["A", "B"], [1, 2])
    got = tr.rating("A").mu

    # brute-force exact Bayes on the same lattice geometry
    grid = tr._grid
    prior = np.exp(-0.5 * (grid / 1.0) ** 2)
    prior /= prior.sum()
    perf_b = np.convolve(prior, kern, mode="same")
    perf_b /= perf_b.sum()
    surv_b = 1.0 - np.cumsum(perf_b)
    # L(a) = sum_x kern(x - a) * surv_b(x), computed directly
    like = np.zeros_like(grid)
    for ai in range(len(grid)):
        acc = 0.0
        for ki in range(len(kern)):
            xi = ai + (ki - half)
            if 0 <= xi < len(grid):
                acc += kern[ki] * surv_b[xi]
            elif xi >= len(grid):
                acc += 0.0
            else:
                acc += kern[ki] * 1.0  # survival is 1 left of the grid
        like[ai] = acc
    post = prior * like
    post /= post.sum()
    exact_mu = -float(np.dot(post, grid))
    assert abs(got - exact_mu) < 5e-3
