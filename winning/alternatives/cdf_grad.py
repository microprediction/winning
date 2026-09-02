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
the honest baseline in comparisons. Note that the estimated vector
sums to one identically for ANY H, so total mass is not a diagnostic;
certify against Monte Carlo argmax frequencies.
"""
import numpy as np

_TINY = 1e-13


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
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    D = np.asarray(D, dtype=float)
    n, k = V.shape
    sd = np.sqrt(D + (V ** 2).sum(1))
    xgrid = jnp.linspace((mu - 7 * sd).min(), (mu + 7 * sd).max(),
                         n_grid)
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
