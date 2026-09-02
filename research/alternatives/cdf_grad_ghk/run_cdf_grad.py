"""Kill test A: all shares from one CDF gradient, factor-state GHK.

The strongest alternative of ALTERNATIVES.md, built honestly: for U = mu + V F +
sqrt(d) eps the rectangle probability H(b) = P(U <= b) is estimated
by GHK with the factor as a Kalman state -- sequential over
alternatives, each step conditioning the k-dimensional factor
posterior on the truncated draw, O(N k^2) per sample with no dense
Cholesky. The full N-vector gradient at the diagonal b = x 1 comes
from one reverse-mode sweep (JAX), and

    p_i = int dH/db_i (x 1) dx

is a trapezoid over a common x-grid with common scrambled Sobol
uniforms across grid points (the smoothness-for-inversion setup).
Complexity O(L R N k^2) for the whole share vector: LINEAR in N --
the construction that could erase the one-orthant-per-alternative
framing.

Measured against the shared-field engine (exact for this covariance
class): total variation, mass defect |sum p - 1|, and wall-clock
(after JIT compilation, which is reported separately). The
Rao-Blackwell prediction from ALTERNATIVES.md: conditional on the
factor path, the GHK draws simulate idiosyncratic shocks the shared
field integrates analytically, so at low rank the adversary pays
Monte Carlo error for nothing. This experiment prices that error.
"""
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS"):
    os.environ.setdefault(_v, "4" if _v != "XLA_FLAGS" else
                          "--xla_cpu_multi_thread_eigen=false")

import jax                                          # noqa: E402
import jax.numpy as jnp                             # noqa: E402
from jax.scipy.special import ndtri                 # noqa: E402
from jax.scipy.stats import norm as jnorm           # noqa: E402
from scipy.stats import qmc                         # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                ".."))
from winning.factor import race_probabilities       # noqa: E402

jax.config.update("jax_enable_x64", True)

R_SOBOL = 512
L_GRID = 96
TINY = 1e-13


def ghk_H_logmean(b, mu, V, d, unifs):
    """log of the GHK estimate of P(U <= b); unifs is (R, N)."""
    k = V.shape[1]

    def one_sample(u):
        def step(carry, inp):
            muF, SigF, logH = carry
            mu_j, v_j, d_j, b_j, u_j = inp
            m = mu_j + v_j @ muF
            Sv = SigF @ v_j
            s2 = v_j @ Sv + d_j
            s = jnp.sqrt(s2)
            e = jnorm.cdf((b_j - m) / s)
            e = jnp.clip(e, TINY, 1.0)
            z = m + s * ndtri(jnp.clip(u_j * e, TINY, 1.0 - 1e-16))
            gain = Sv / s2
            muF2 = muF + gain * (z - m)
            SigF2 = SigF - jnp.outer(gain, Sv)
            return (muF2, SigF2, logH + jnp.log(e)), None

        init = (jnp.zeros(k), jnp.eye(k), 0.0)
        carry, _ = jax.lax.scan(step, init, (mu, V, d, b, u))
        return carry[2]

    logs = jax.vmap(one_sample)(unifs)
    return jax.scipy.special.logsumexp(logs) - jnp.log(unifs.shape[0])


def build(mu, V, d, unifs, xgrid):
    """Returns jitted p-hat over the grid via reverse-mode at the
    diagonal."""
    mu_j = jnp.asarray(mu)
    V_j = jnp.asarray(V)
    d_j = jnp.asarray(d)
    u_j = jnp.asarray(unifs)
    ones = jnp.ones_like(mu_j)

    def H_of_b(b):
        return jnp.exp(ghk_H_logmean(b, mu_j, V_j, d_j, u_j))

    grad_H = jax.grad(H_of_b)

    def shares(xs):
        # sequential over grid points: reverse-mode stores the scan
        # residuals for one x at a time, bounding memory at O(R N k^2)
        g = jax.lax.map(lambda x: grad_H(x * ones), xs)  # (L, N)
        w = jnp.gradient(xs)                             # trapezoid
        return (g * w[:, None]).sum(0)

    return jax.jit(shares)


def make_instance(n, k, share, rng):
    mu = rng.normal(0.0, 1.0, n)
    u = rng.normal(size=(n, k))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    base = 0.3 + rng.random(n)
    V = u * np.sqrt(share * base)[:, None]
    d = (1.0 - share) * base
    return mu, V, d


if __name__ == "__main__":
    results = {}
    for n, k in ((50, 2), (200, 2), (1000, 2), (200, 4)):
        rng = np.random.default_rng(n + k)
        mu, V, d = make_instance(n, k, 0.5, rng)
        t0 = time.time()
        # the CDF gradient prices the MAX winner; the engine is
        # min-wins, so negate
        p_exact = race_probabilities(-mu, V=-V, D=d)
        t_exact = time.time() - t0
        sd = np.sqrt(d + (V ** 2).sum(1))
        lo = (mu - 7 * sd).min()
        hi = (mu + 7 * sd).max()
        xgrid = jnp.linspace(lo, hi, L_GRID)
        sob = qmc.Sobol(d=n, scramble=True, seed=5)
        unifs = sob.random(R_SOBOL)
        f = build(mu, V, d, unifs, xgrid)
        t0 = time.time()
        p_hat = np.asarray(f(xgrid))
        t_compile_and_run = time.time() - t0
        t0 = time.time()
        p_hat = np.asarray(f(xgrid))
        t_run = time.time() - t0
        tv = 0.5 * np.abs(p_hat - p_exact).sum()
        mass = float(p_hat.sum())
        print(f"[N={n} k={k}] ghk-grad tv {tv:.4f} mass {mass:.4f} "
              f"run {t_run:.2f}s (compile+run {t_compile_and_run:.1f}s)"
              f" | exact {t_exact:.3f}s")
        results[f"n{n}_k{k}"] = dict(
            tv=float(tv), mass=mass, seconds=t_run,
            compile_seconds=t_compile_and_run,
            exact_seconds=t_exact, R=R_SOBOL, L=L_GRID)
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
