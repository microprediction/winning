"""The published fastrace extension is its own API: direct callers never
pass through winning's Python normalization. These tests pin the PyO3
boundary itself -- GIL policy, fork safety, and that malformed input is a
ValueError, never a pyo3 PanicException. Skipped if fastrace is absent."""
import sys
import threading
import time

import numpy as np
import pytest

fastrace = pytest.importorskip("fastrace")


def _classic_inputs(n_offsets=3000):
    x = np.arange(2001) - 1000
    d = np.exp(-0.5 * (x / 80.0) ** 2)
    d /= d.sum()
    return d.tolist(), np.linspace(-100, 100, n_offsets).tolist()


# --- #120: classic kernels release the GIL --------------------------------

def _background_progress(call):
    """Counter increments a Python thread makes while `call` runs."""
    count = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            count[0] += 1

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    t = threading.Thread(target=spin)
    try:
        t.start()
        time.sleep(0.05)
        before = count[0]
        t0 = time.perf_counter()
        call()
        elapsed = time.perf_counter() - t0
        delta = count[0] - before
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old)
    return delta, elapsed


@pytest.mark.parametrize("kernel", ["state_prices", "calibrate"])
def test_classic_kernels_release_the_gil(kernel):
    d, offsets = _classic_inputs()
    if kernel == "state_prices":
        call = lambda: fastrace.classic_exact_state_prices(d, offsets)
    else:
        prices = list(np.linspace(1.0, 2.0, 400) / 600.0)
        samples = list(np.linspace(100, -100, 400))
        guess = list(np.linspace(-50, 50, 400))
        call = lambda: fastrace.classic_exact_calibrate(d, prices, samples, guess, 3)
    # calibrate a reference rate of the spinner over the same wall time
    delta, elapsed = _background_progress(call)
    ref, ref_elapsed = _background_progress(lambda: time.sleep(elapsed))
    rate, ref_rate = delta / elapsed, ref / ref_elapsed
    # holding the GIL measured ~1% of the free rate; released is ~50-100%
    assert rate > 0.2 * ref_rate, (rate, ref_rate, elapsed)


# --- #122: a classic-first warmup must not defeat the fork guard -----------

@pytest.mark.skipif(sys.platform == "win32", reason="fork is POSIX")
def test_classic_first_warmup_then_fork_does_not_deadlock():
    script = r'''
import multiprocessing as mp, numpy as np, sys
import fastrace
x = np.arange(1001) - 500
d = np.exp(-0.5 * (x / 80.0) ** 2); d /= d.sum()
fastrace.classic_exact_state_prices(d.tolist(), np.linspace(-100, 100, 200).tolist())
def child(q):
    mu = np.linspace(-2, 2, 120); sd = np.ones(120)
    q.put(float(np.sum(fastrace.top_k(mu, sd, 40, -12.0, 12.0, 513))))
if __name__ == "__main__":
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start(); p.join(20)
    if p.is_alive():
        p.terminate(); p.join(); sys.exit(3)
    sys.exit(0 if abs(q.get(timeout=5) - 40.0) < 1e-6 else 4)
'''
    import subprocess
    # a fresh interpreter, so the classic call is genuinely the first
    # rayon region the parent ever enters
    r = subprocess.run([sys.executable, "-c", script], timeout=120)
    assert r.returncode == 0, "forked child hung (3) or mispriced (4)"


# --- #74: cross-array shapes are ValueErrors, never PanicExceptions -------

def _factor_ok(n=4, r=1, q=3):
    mu = np.arange(n, dtype=float)
    v = np.zeros((n, r))
    d = np.ones(n)
    f = np.zeros((q, r))
    w = np.full(q, 1.0 / q)
    return mu, v, d, f, w


def _factor_bad_cases(n=4):
    mu, v, d, f, w = _factor_ok(n)
    yield "v rows", (mu, np.ones((1, n)), d, np.zeros((3, n)), w)
    yield "d len", (mu, v, np.ones(n - 1), f, w)
    yield "f cols", (mu, v, d, np.zeros((3, 2)), w)
    yield "w len", (mu, v, d, f, np.full(2, 0.5))
    yield "empty", (mu[:0], v[:0], d[:0], f, w)


FACTOR_CALLS = {
    "forward_and_slopes": lambda a: fastrace.forward_and_slopes(*a, 129),
    "win_probabilities_factor":
        lambda a: fastrace.win_probabilities_factor(*a, 129),
    "win_probabilities_factor_separated":
        lambda a: fastrace.win_probabilities_factor_separated(*a, 129),
    "ordered_prefixes": lambda a: fastrace.ordered_prefixes(*a, 129, k=2),
    "jacobian_vector_product":
        lambda a: fastrace.jacobian_vector_product(*a, np.ones(len(a[0])),
                                                   129),
    "forward_and_slopes_base":
        lambda a: fastrace.forward_and_slopes_base(*a, 129, np.nan, np.nan,
                                                   0, []),
}


@pytest.mark.parametrize("name", sorted(FACTOR_CALLS))
def test_factor_kernels_reject_mismatched_shapes(name):
    call = FACTOR_CALLS[name]
    call(_factor_ok())                      # the valid shapes still run
    for label, args in _factor_bad_cases():
        with pytest.raises(ValueError):
            call(args)
    with pytest.raises(ValueError, match="points"):
        a = _factor_ok()
        if name == "jacobian_vector_product":
            fastrace.jacobian_vector_product(*a, np.ones(4), 1)
        elif name == "forward_and_slopes_base":
            fastrace.forward_and_slopes_base(*a, 1, np.nan, np.nan, 0, [])
        else:
            getattr(fastrace, name)(*a, 1)


def test_jvp_direction_length_is_checked():
    mu, v, d, f, w = _factor_ok()
    with pytest.raises(ValueError, match="h has length 3"):
        fastrace.jacobian_vector_product(mu, v, d, f, w, np.ones(3), 129)


def test_public_jvp_same_exception_class_with_and_without_rust():
    import winning
    from winning.factor.core import hermite_nodes, jacobian_vector_product
    mu = np.arange(4., dtype=float)
    V = np.zeros((4, 1))
    D = np.ones(4)
    F, W = hermite_nodes(1, Q=3)
    h = np.ones(3)
    try:
        for flag in (True, False):
            winning.use_rust(flag)
            with pytest.raises(ValueError):
                jacobian_vector_product(mu, V, D, F, W, h, points=101)
    finally:
        winning.use_rust(True)


def test_unknown_base_id_is_a_value_error():
    with pytest.raises(ValueError, match="base_id"):
        fastrace.forward_and_slopes_base(*_factor_ok(), 129, np.nan, np.nan,
                                         99, [])


def test_ghk_and_per_winner_shapes():
    mu, v, d, _, _ = _factor_ok()
    fastrace.ghk_all_shares(mu, v, d, 50)
    with pytest.raises(ValueError):
        fastrace.ghk_all_shares(mu, np.ones((1, 4)), d, 50)
    with pytest.raises(ValueError):
        fastrace.ghk_all_shares(mu, v, d[:3], 50)
    z = np.zeros((10, 2))
    fastrace.per_winner_reduced_rank(mu, v, d, z)
    with pytest.raises(ValueError, match="z has 1 columns"):
        fastrace.per_winner_reduced_rank(mu, v, d, z[:, :1])
    with pytest.raises(ValueError):
        fastrace.per_winner_reduced_rank(mu, v, d[:2], z)


def _block_ok(n=4):
    mu = np.linspace(-1, 1, n)
    sd = np.ones(n)
    v = np.full(n, 0.3)
    starts = np.array([0, 2], dtype=np.int64)
    an = np.array([-1.0, 0.0, 1.0])
    aw = np.array([0.25, 0.5, 0.25])
    return mu, sd, v, starts, an, aw


def test_block_race_shapes_and_starts():
    mu, sd, v, starts, an, aw = _block_ok()
    fastrace.block_race(mu, sd, v, starts, an, aw, 129)
    bad = [
        (mu, sd[:3], v, starts, an, aw),
        (mu, sd, v[:3], starts, an, aw),
        (mu, sd, v, starts, an, aw[:2]),
        (mu, sd, v, np.array([1, 2], dtype=np.int64), an, aw),
        (mu, sd, v, np.array([0, -1], dtype=np.int64), an, aw),
        (mu, sd, v, np.array([0, 9], dtype=np.int64), an, aw),
        (mu, sd, v, np.array([0, 3, 1], dtype=np.int64), an, aw),
        (mu, sd, v, np.array([], dtype=np.int64), an, aw),
    ]
    for args in bad:
        with pytest.raises(ValueError):
            fastrace.block_race(*args, 129)
    vr = v[:, None]
    nd = an[:, None]
    fastrace.block_race_r(mu, sd, vr, starts, nd, aw, 129)
    for args in [
        (mu, sd, np.ones((3, 1)), starts, nd, aw),
        (mu, sd, vr, starts, np.ones((3, 2)), aw),
        (mu, sd, vr, starts, nd, aw[:2]),
        (mu, sd, vr, np.array([0, 7], dtype=np.int64), nd, aw),
    ]:
        with pytest.raises(ValueError):
            fastrace.block_race_r(*args, 129)


def test_tree_race_shapes_and_topology():
    mu, sd, v, starts, an, aw = _block_ok()
    parent = np.array([2, 2, -1], dtype=np.int64)   # two leaves, one root
    lam = np.array([0.0, 0.0, 0.3])
    fastrace.tree_race(mu, sd, v, starts, parent, lam, an, aw, 129, -8., 8.)
    for pa, lm in [
        (np.array([2, 2, 5], dtype=np.int64), lam),       # out of range
        (np.array([1, 0, -1], dtype=np.int64), lam),      # cycle
        (np.array([2, 2, 2], dtype=np.int64), lam),       # self-parent
        (np.array([-1], dtype=np.int64), lam[:1]),        # fewer than leaves
        (parent, lam[:2]),                                # lam length
    ]:
        with pytest.raises(ValueError):
            fastrace.tree_race(mu, sd, v, starts, pa, lm, an, aw, 129,
                               -8., 8.)


@pytest.mark.parametrize("name", ["top_k", "top_k_slopes", "top_k_jacobians"])
def test_topk_mu_sd_lengths(name):
    with pytest.raises(ValueError, match="sd has length 2"):
        getattr(fastrace, name)(np.zeros(3), np.ones(2), 1, -8., 8., 129)


def test_rank_and_window_mu_sd_lengths():
    with pytest.raises(ValueError):
        fastrace.rank_marginals(np.zeros(3), np.ones(2), -8., 8., 129)
    with pytest.raises(ValueError):
        fastrace.rank_marginal_jacobian(np.zeros(3), np.ones(2), 1, -8., 8.,
                                        129)
    with pytest.raises(ValueError):
        fastrace.top_k_window(np.zeros(3), np.ones(2), 1)


# --- #403: ordered_prefixes supports only k in {1, 2, 3} ------------------

def test_ordered_prefixes_rejects_unsupported_k():
    mu = np.array([0.0, 0.2, 0.7])
    args = (mu, np.zeros((3, 1)), np.ones(3), np.zeros((1, 1)), np.ones(1),
            1001, -8.0, 8.7)
    flat3, total3 = fastrace.ordered_prefixes(*args, 3)
    assert np.asarray(flat3).size == 27 and abs(total3 - 1) < 1e-4
    for k in (1, 2):
        flat, total = fastrace.ordered_prefixes(*args, k)
        assert np.asarray(flat).size == 3 ** k and abs(total - 1) < 1e-4
    for k in (0, 4, 5):
        with pytest.raises(ValueError, match="1, 2 or 3"):
            fastrace.ordered_prefixes(*args, k)


# --- #436: top-k depths in [1, n-1], exact ranks in [1, n] ----------------

TOPK_DEPTH_CALLS = {
    "top_k": lambda mu, sd, k: fastrace.top_k(mu, sd, k, -8., 9., 1001),
    "top_k_slopes":
        lambda mu, sd, k: fastrace.top_k_slopes(mu, sd, k, -8., 9., 1001),
    "top_k_jacobians":
        lambda mu, sd, k: fastrace.top_k_jacobians(mu, sd, k, -8., 9., 1001),
    "top_k_window": lambda mu, sd, k: fastrace.top_k_window(mu, sd, k),
}


@pytest.mark.parametrize("name", sorted(TOPK_DEPTH_CALLS))
def test_topk_depth_contract(name):
    call = TOPK_DEPTH_CALLS[name]
    mu, sd = np.array([0.0, 0.2, 0.7]), np.ones(3)
    for k in (1, 2):                                # valid endpoints
        call(mu, sd, k)
    for k in (0, 3, 4):                             # 0, n, n+1
        with pytest.raises(ValueError, match=r"\[1, n-1\]"):
            call(mu, sd, k)


def test_topk_valid_depths_have_mass_k():
    mu, sd = np.array([0.0, 0.2, 0.7]), np.ones(3)
    for k in (1, 2):
        q = np.asarray(fastrace.top_k(mu, sd, k, -8., 9., 1001))
        assert abs(q.sum() - k) < 1e-6


def test_rank_marginal_jacobian_rank_contract():
    mu, sd = np.array([0.0, 0.2, 0.7]), np.ones(3)
    total = np.zeros(3)
    for r in (1, 2, 3):                             # valid endpoints
        p, _ = fastrace.rank_marginal_jacobian(mu, sd, r, -8., 9., 1001)
        total += np.asarray(p)
    assert np.allclose(total, 1.0, atol=1e-6)
    for r in (0, 4):
        with pytest.raises(ValueError, match=r"one-based"):
            fastrace.rank_marginal_jacobian(mu, sd, r, -8., 9., 1001)


def test_one_runner_rank_window_still_allowed():
    # the rank kernels ask for the window at depth n-1, which is 0 here
    fastrace.top_k_window(np.zeros(1), np.ones(1), 0)


# --- #218: the separated Chebyshev pass is unit-free -----------------------

@pytest.mark.parametrize("spread", [False, True])
def test_separated_kernel_scale_invariance(spread):
    rel = np.array([1.0, 1.5, 0.7]) if spread else np.ones(3)

    def price(c):
        return np.asarray(fastrace.win_probabilities_factor_separated(
            c * np.array([0.0, 0.4, 1.0]), np.zeros((3, 1)), c * c * rel,
            np.zeros((1, 1)), np.ones(1), points=1501, rm=48, rs=14)[0])

    p1 = price(1.0)
    if not spread:
        assert np.allclose(p1, [0.5289947793, 0.3279161766, 0.1430890441],
                           atol=1e-6)
    for e in range(-8, 9):
        pc = price(10.0 ** e)
        assert np.all(np.isfinite(pc)), e
        assert np.max(np.abs(pc - p1)) < 1e-9, (e, pc, p1)


# --- #506 #516 #519 #522 #602: values are checked, not only shapes --------
# Signed scales, reversed windows, NaN factor nodes and zero Chebyshev
# orders used to reach the lattices and return finite but impossible
# answers (negative "probabilities", or the law of a different model).

BAD_SCALES = [0.0, -1.0, np.nan, np.inf]
TOPK_MU = np.array([0.0, 0.2, 0.7])

TOPK_VALUE_CALLS = {
    "top_k": lambda mu, sd, lo, hi: fastrace.top_k(mu, sd, 1, lo, hi, 1001),
    "top_k_slopes":
        lambda mu, sd, lo, hi: fastrace.top_k_slopes(mu, sd, 1, lo, hi, 1001),
    "top_k_jacobians":
        lambda mu, sd, lo, hi: fastrace.top_k_jacobians(mu, sd, 1, lo, hi,
                                                        1001),
    "rank_marginals":
        lambda mu, sd, lo, hi: fastrace.rank_marginals(mu, sd, lo, hi, 1001),
    "rank_marginal_jacobian":
        lambda mu, sd, lo, hi: fastrace.rank_marginal_jacobian(mu, sd, 1, lo,
                                                               hi, 1001),
}


@pytest.mark.parametrize("name", sorted(TOPK_VALUE_CALLS))
def test_topk_rank_reject_bad_scales_and_windows(name):
    call = TOPK_VALUE_CALLS[name]
    call(TOPK_MU, np.ones(3), -8.0, 9.0)                    # valid control
    for s in BAD_SCALES:                                    # #506
        with pytest.raises(ValueError, match=r"sd\[1\]"):
            call(TOPK_MU, np.array([1.0, s, 1.0]), -8.0, 9.0)
    with pytest.raises(ValueError, match=r"mu\[0\]"):
        call(np.array([np.nan, 0.2, 0.7]), np.ones(3), -8.0, 9.0)
    for lo, hi in [(9.0, -8.0), (0.0, 0.0), (np.nan, 9.0), (-8.0, np.inf),
                   (-np.inf, 9.0)]:                         # #516
        with pytest.raises(ValueError, match="window"):
            call(TOPK_MU, np.ones(3), lo, hi)


def test_topk_reversed_window_used_to_negate_mass():
    q = np.asarray(fastrace.top_k(TOPK_MU, np.ones(3), 1, -8.0, 9.0, 1001))
    assert np.all(q > 0) and abs(q.sum() - 1) < 1e-9


def test_top_k_window_controls_and_scales():
    lo, hi = fastrace.top_k_window(TOPK_MU, np.ones(3), 1)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
    for s in BAD_SCALES:
        with pytest.raises(ValueError, match=r"sd\[1\]"):
            fastrace.top_k_window(TOPK_MU, np.array([1.0, s, 1.0]), 1)
    for delta, pad in [(1.0, 2.0), (np.inf, 2.0), (0.0, 2.0), (np.nan, 2.0),
                       (1e-12, -2.0), (1e-12, np.nan), (1e-12, np.inf)]:
        with pytest.raises(ValueError, match="delta|pad_sds"):
            fastrace.top_k_window(TOPK_MU, np.ones(3), 1, delta, pad)


@pytest.mark.parametrize("name", sorted(FACTOR_CALLS))
def test_factor_kernels_reject_bad_values(name):
    call = FACTOR_CALLS[name]
    for s in BAD_SCALES:                                    # #519
        mu, v, d, f, w = _factor_ok()
        d[1] = s
        with pytest.raises(ValueError, match=r"d\[1\]"):
            call((mu, v, d, f, w))
    for which, idx in [("mu", 2), ("v", (1, 0)), ("f", (1, 0)),
                       ("w", 0)]:                           # #602
        for bad in (np.nan, np.inf):
            a = list(_factor_ok())
            pos = "mu v d f w".split().index(which)
            a[pos] = a[pos].copy()
            a[pos][idx] = bad
            with pytest.raises(ValueError, match=rf"{which}\["):
                call(tuple(a))


def test_nan_factor_node_no_longer_prices_another_law():
    # #602: the bad two-node rule returned exactly the one-node law
    mu = np.array([0.0, 0.3, 1.0])
    V = np.array([[0.0], [1.0], [-0.5]])
    good = fastrace.win_probabilities_factor(
        mu, V, np.ones(3), np.array([[-5.0], [5.0]]), np.array([.5, .5]), 501)
    assert abs(good[1] - 1) < 1e-9
    with pytest.raises(ValueError, match=r"f\[1, 0\] = NaN"):
        fastrace.win_probabilities_factor(
            mu, V, np.ones(3), np.array([[-5.0], [np.nan]]),
            np.array([.5, .5]), 501)


def test_factor_jvp_direction_must_be_finite():
    mu, v, d, f, w = _factor_ok()
    with pytest.raises(ValueError, match=r"h\[1\]"):
        fastrace.jacobian_vector_product(mu, v, d, f, w,
                                         np.array([0., np.nan, 0., 0.]), 129)


@pytest.mark.parametrize("name", ["forward_and_slopes",
                                  "win_probabilities_factor",
                                  "ordered_prefixes"])
def test_factor_explicit_windows_are_checked(name):
    a = _factor_ok()
    extra = {"k": 2} if name == "ordered_prefixes" else {}
    call = getattr(fastrace, name)
    call(*a, 129, np.nan, np.nan, **extra)                  # automatic
    call(*a, 129, -9.0, 12.0, **extra)                      # explicit
    for lo, hi in [(12.0, -9.0), (1.0, 1.0), (np.nan, 12.0), (-9.0, np.inf)]:
        with pytest.raises(ValueError, match="window"):
            call(*a, 129, lo, hi, **extra)


def test_separated_kernel_orders_must_be_positive():
    # #522: rm = 0 or rs = 0 panicked (heterogeneous D) or was silently
    # replaced by 1 (homogeneous D)
    mu = np.array([0.0, 0.4, 1.0])
    V, F, W = np.zeros((3, 1)), np.zeros((2, 1)), np.array([0.5, 0.5])
    for D in (np.ones(3), np.array([1.0, 2.0, 3.0])):
        fastrace.win_probabilities_factor_separated(mu, V, D, F, W,
                                                    points=101, rm=1, rs=1)
        for rm, rs in [(0, 14), (48, 0), (0, 0)]:
            with pytest.raises(ValueError, match="Chebyshev orders"):
                fastrace.win_probabilities_factor_separated(
                    mu, V, D, F, W, points=101, rm=rm, rs=rs)


def test_structured_kernels_reject_bad_values():
    mu, sd, v, starts, an, aw = _block_ok()
    parent = np.array([2, 2, -1], dtype=np.int64)
    lam = np.array([0.0, 0.0, 0.3])
    calls = {
        "block_race": lambda mu, sd, v, an, aw, lo=np.nan, hi=np.nan:
            fastrace.block_race(mu, sd, v, starts, an, aw, 129, lo, hi),
        "block_race_r": lambda mu, sd, v, an, aw, lo=np.nan, hi=np.nan:
            fastrace.block_race_r(mu, sd, v[:, None], starts, an[:, None],
                                  aw, 129, lo, hi),
        "tree_race": lambda mu, sd, v, an, aw, lo=-8.0, hi=8.0:
            fastrace.tree_race(mu, sd, v, starts, parent, lam, an, aw, 129,
                               lo, hi),
    }
    for name, call in calls.items():
        p = np.asarray(call(mu, sd, v, an, aw))             # valid control
        assert abs(p.sum() - 1) < 1e-6, name
        for s in BAD_SCALES:                                # #519
            bad = sd.copy(); bad[1] = s
            with pytest.raises(ValueError, match=r"sd\[1\]"):
                call(mu, bad, v, an, aw)
        for arg in range(5):                                # #602 family
            if arg == 1:
                continue
            a = [mu.copy(), sd, v.copy(), an.copy(), aw.copy()]
            a[arg][0] = np.nan
            with pytest.raises(ValueError, match="not finite"):
                call(*a)
        for lo, hi in [(8.0, -8.0), (np.nan, 8.0), (-8.0, np.inf)]:  # #516
            with pytest.raises(ValueError, match="window"):
                call(mu, sd, v, an, aw, lo, hi)
    with pytest.raises(ValueError, match="window"):
        calls["tree_race"](mu, sd, v, an, aw, np.nan, np.nan)


def test_per_winner_reduced_rank_rejects_bad_values():
    mu, v, d, _, _ = _factor_ok()
    z = np.zeros((10, 2))
    for s in BAD_SCALES:
        dd = d.copy(); dd[1] = s
        with pytest.raises(ValueError, match=r"d\[1\]"):
            fastrace.per_winner_reduced_rank(mu, v, dd, z)
    zz = z.copy(); zz[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"z\[3, 1\]"):
        fastrace.per_winner_reduced_rank(mu, v, d, zz)


# --- #526: an empty classic field is a ValueError, not a panic ------------

CLASSIC_DENSITY = [0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]


def test_classic_empty_field_is_a_value_error():
    assert len(fastrace.classic_exact_state_prices(CLASSIC_DENSITY,
                                                   [0.0])) == 1
    with pytest.raises(ValueError, match="offsets is empty"):
        fastrace.classic_exact_state_prices(CLASSIC_DENSITY, [])
    for n_iter in (0, 1, 3):
        with pytest.raises(ValueError, match="prices is empty"):
            fastrace.classic_exact_calibrate(CLASSIC_DENSITY, [],
                                             [1.0, 0.0, -1.0], [], n_iter)
    with pytest.raises(ValueError, match="not finite"):
        fastrace.classic_exact_state_prices(CLASSIC_DENSITY, [0.0, np.nan])


# --- #517: a tree has exactly one root ------------------------------------

def test_tree_race_rejects_a_forest():
    mu = np.array([0.0, 1.0])
    starts = np.array([0, 1], dtype=np.int64)
    args = (-mu, np.ones(2), np.zeros(2), starts)
    with pytest.raises(ValueError, match="exactly one root.*got 2"):
        fastrace.tree_race(*args, np.array([-1, -1], dtype=np.int64),
                           np.zeros(2), np.array([0.0]), np.ones(1),
                           257, -9.0, 10.0)
    # the zero-strength common root is the independent race
    p = np.asarray(fastrace.tree_race(
        *args, np.array([2, 2, -1], dtype=np.int64), np.zeros(3),
        np.array([0.0]), np.ones(1), 257, -9.0, 10.0))
    assert abs(p.sum() - 1) < 1e-6 and p[0] > p[1]


# --- #473: the calibration offset grid must descend -----------------------

def test_classic_calibrate_requires_descending_offsets():
    from winning.classic.lattice import skew_normal_density
    d = list(skew_normal_density(50, 0.1))
    target = fastrace.classic_exact_state_prices(d, [-3.0, 0.5, 2.0])
    desc = [float(x) for x in range(24, -26, -1)]
    a = fastrace.classic_exact_calibrate(d, target, desc, [0.0] * 3, 3)
    back = fastrace.classic_exact_state_prices(d, a)
    assert max(abs(x - y) for x, y in zip(back, target)) < 1e-4
    for bad in (desc[::-1], desc[:10] + [30.0] + desc[10:],
                desc[:5] + [np.nan] + desc[5:]):
        with pytest.raises(ValueError, match="offset_samples"):
            fastrace.classic_exact_calibrate(d, target, bad, [0.0] * 3, 3)


# --- #601: the separated pass never returns an impossible law -------------

def test_separated_kernel_falls_back_outside_its_regime():
    mu = np.array([0.4, -1.0, -0.6])
    V, F, W = np.zeros((3, 1)), np.zeros((1, 1)), np.ones(1)
    D = np.array([1e-4, 0.16, 0.16])
    p, total = fastrace.win_probabilities_factor_separated(
        mu, V, D, F, W, points=1501, rm=48, rs=14)
    p = np.asarray(p)
    ref, _ = fastrace.win_probabilities_factor(mu, V, D, F, W, 1501)
    assert np.all(p >= 0) and np.all(p <= 1)
    assert np.max(np.abs(p - np.asarray(ref))) < 1e-9
    assert abs(total - 1) < 1e-6
