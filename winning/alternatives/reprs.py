"""Reduced-rank rectangle representations of single-winner events."""
import numpy as np


def reduced_rank_representation(mu, V, D, i):
    """Winner i's probability as a reduced-rank Gaussian rectangle.

    Max-wins: i beats the field iff Y_j = U_j - U_i <= 0 for all
    j != i, and with U = mu + V F + sqrt(D) eps the difference vector
    has covariance B B' + diag(D_minus) where B carries k+1 columns:
    the loading differences and the shared -sqrt(D_i) column from the
    winner's own shock. The identity is Marsaglia's (1963), developed
    by Genz and Bretz, and is the representation mvtnorm::lpRR
    consumes; one call per winner prices the full vector at O(R N^2),
    which is what the shared field's single O(QNL) pass replaces.

    Returns a dict with keys B (n-1, k+1), D_minus (n-1,), upper
    (n-1,), so that P(i wins) = P(N(0, B B' + diag(D_minus)) <= upper).
    """
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    D = np.asarray(D, dtype=float)
    mask = np.arange(len(mu)) != i
    B = np.column_stack([V[mask] - V[i], -np.sqrt(D[i])
                         * np.ones(mask.sum())])
    return dict(B=B, D_minus=D[mask], upper=mu[i] - mu[mask])


def per_winner_reduced_rank_shares(mu, V, D, n_samples=512, seed=11):
    """The complete max-wins share vector by the per-winner protocol:
    one reduced-rank rectangle per winner over common scrambled-Sobol
    draws -- the lpRR benchmark alternative, O(N^2 R k) by design.
    Dispatches to the compiled kernel when fastrace is importable and
    falls back to numpy; both normalize the vector. The engine's
    shared field computes the same vector exactly in O(NLQ); this
    exists so that comparison can be rerun same-toolchain."""
    from scipy.stats import norm, qmc
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    D = np.asarray(D, dtype=float)
    n, k = V.shape
    Z = norm.ppf(qmc.Sobol(d=k + 1, scramble=True,
                           seed=seed).random(n_samples))
    try:
        import fastrace
        p = fastrace.per_winner_reduced_rank(mu, V, D, Z)
    except (ImportError, AttributeError):
        sd = np.sqrt(D)
        p = np.empty(n)
        for i in range(n):
            bz = (V - V[i]) @ Z[:, :k].T            # (n, R)
            arg = (mu[i] - mu[:, None] - bz
                   + sd[i] * Z[None, :, k]) / sd[:, None]
            logp = np.log(np.clip(norm.cdf(arg), 1e-300, 1.0))
            logp[i] = 0.0
            p[i] = np.exp(logp.sum(0)).mean()
    return p / p.sum()
