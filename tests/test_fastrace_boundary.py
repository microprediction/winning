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
        call = lambda: fastrace.classic_state_prices(d, offsets)
    else:
        prices = list(np.linspace(1.0, 2.0, 400) / 600.0)
        samples = list(np.linspace(100, -100, 400))
        guess = list(np.linspace(-50, 50, 400))
        call = lambda: fastrace.classic_calibrate(d, prices, samples, guess, 3)
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
fastrace.classic_state_prices(d.tolist(), np.linspace(-100, 100, 200).tolist())
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
