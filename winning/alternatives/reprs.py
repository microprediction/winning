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
