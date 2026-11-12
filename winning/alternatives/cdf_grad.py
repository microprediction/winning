"""All winner probabilities from one CDF gradient, by factor-state GHK.

The identity: with H(b) = P(U <= b) and M = max_i U_i,
p_i = int dH/db_i (x 1) dx -- every winner probability is the
integrated boundary gradient of the maximum's CDF along the diagonal.
This module estimates H by GHK with the factors as a Kalman state
(O(N k^2) per sample, no dense Cholesky) and takes the full N-vector
gradient in one reverse-mode sweep per grid point, so the whole share
vector is linear in N. Requires jax.

Within the factor grammar the shared-field engine dominates this
evaluator at every rank and accuracy we have measured (see
research/alternatives/cdf_grad_ghk/): conditional on the factors the
GHK draws simulate idiosyncratic dimensions the field integrates
analytically. Use this evaluator to cross-check the engine with
independent machinery, as a starting point for GPU experiments, or as
the honest baseline in comparisons. The exact integral sums to one for
ANY H, so total mass is not a certificate; certify against Monte Carlo
argmax frequencies. The finite rectangle sum does not: a runner whose
boundary-gradient spike falls between grid points is lost outright, so
the grid carries a local band for every runner the global spacing cannot
resolve (#955).
"""
import numpy as np

from ..shapes import as_idio, as_loadings


_TINY = 1e-13


# points across one runner's +-7 sd band below which the global grid is
# taken not to resolve it, and the size of the local band then added
_RESOLVE = 16
_LOCAL = 33


def _union_grid(mu, sd, n_grid):
    """One global grid over every marginal's +-7 sd, plus a local band for
    each runner narrower than _RESOLVE global spacings. A uniform grid
    sized by the widest marginal put 0.001-sd runner 0 of a 1e-6/1/100
    variance field between every point: its share came back exactly zero
    and the vector lost 18% of its mass (#955). Bands are added only where
    needed, so a balanced field keeps the global grid unchanged."""
    lo, hi = float((mu - 7 * sd).min()), float((mu + 7 * sd).max())
    grid = [np.linspace(lo, hi, n_grid)]
    h = (hi - lo) / (n_grid - 1)
    for m, s in zip(mu, sd):
        if 14 * s < _RESOLVE * h:
            grid.append(np.linspace(m - 7 * s, m + 7 * s, _LOCAL))
    return np.unique(np.concatenate(grid)) if len(grid) > 1 else grid[0]


def cdf_gradient_shares(mu, V, D, n_samples=512, n_grid=96, seed=5):
    """Max-wins winner probabilities, all N at once. Requires jax."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import ndtri
        from jax.scipy.stats import norm as jnorm
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "cdf_gradient_shares requires jax (pip install jax); the "
            "shared-field engine in winning.factor needs no such "
            "dependency and is faster within the factor grammar"
        ) from exc
    from scipy.stats import qmc

    jax.config.update("jax_enable_x64", True)
    mu = np.asarray(mu, dtype=float)
    V = as_loadings(V, len(np.asarray(mu)))
    # the central contract: a scalar is everyone's variance, a short or
    # negative D is refused (it broadcast or reached the scan; #955 review,
    # invisible to CI, which has no jax)
    D = as_idio(D, len(mu))
    n, k = V.shape
    sd = np.sqrt(D + (V ** 2).sum(1))
    xgrid = jnp.asarray(_union_grid(mu, sd, n_grid))
    unifs = jnp.asarray(qmc.Sobol(d=n, scramble=True,
                                  seed=seed).random(n_samples))
    mu_j, V_j, d_j = jnp.asarray(mu), jnp.asarray(V), jnp.asarray(D)
    ones = jnp.ones(n)

    def one_sample(b, u):
        def step(carry, inp):
            muF, SigF, logH = carry
            m_i, v_i, dd, b_i, u_i = inp
            m = m_i + v_i @ muF
            Sv = SigF @ v_i
            s2 = v_i @ Sv + dd
            s = jnp.sqrt(s2)
            e = jnp.clip(jnorm.cdf((b_i - m) / s), _TINY, 1.0)
            z = m + s * ndtri(jnp.clip(u_i * e, _TINY, 1 - 1e-16))
            gain = Sv / s2
            return (muF + gain * (z - m), SigF - jnp.outer(gain, Sv),
                    logH + jnp.log(e)), None

        init = (jnp.zeros(k), jnp.eye(k), 0.0)
        carry, _ = jax.lax.scan(step, init, (mu_j, V_j, d_j, b, u))
        return carry[2]

    def H_of_b(b):
        logs = jax.vmap(lambda u: one_sample(b, u))(unifs)
        return jnp.exp(jax.scipy.special.logsumexp(logs)
                       - jnp.log(unifs.shape[0]))

    grad_H = jax.grad(H_of_b)

    @jax.jit
    def shares(xs):
        g = jax.lax.map(lambda x: grad_H(x * ones), xs)
        w = jnp.gradient(xs)
        return (g * w[:, None]).sum(0)

    return np.asarray(shares(xgrid))
