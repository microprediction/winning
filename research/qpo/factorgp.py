"""Phase VIII: get (mu, V, D) out of the GP without ever forming Sigma.

The eigen-oracle used in phases II-VII is deliberately not an algorithm: it
needs the N x N posterior covariance, which is the thing that forces the qPO
implementation to prefilter to 10,000 candidates in the first place. This
module produces the same kind of object -- a rank-r factor model with the
exact GP marginal variances on the diagonal -- in time linear in N.

The exact Tanimoto GP posterior over candidates X*, given n observations, is

    mu_*    = m0 + s^2 K_*n A^{-1} (y - m0),        A = s^2 K_nn + sigma^2 I
    Sigma_* = s^2 K_** - s^4 K_*n A^{-1} K_n* + sigma^2 I

with s^2 the outputscale, sigma^2 the likelihood noise, m0 the constant mean.
Two of those three terms are already cheap: the data correction has rank n
(one hundred, here) and the noise is diagonal. Only the prior block s^2 K_**
is genuinely N x N, and a Nystrom approximation on r_z inducing molecules Z
handles it:

    K_** ~= K_*Z K_ZZ^{-1} K_Z*  =  Phi Phi',   Phi = K_*Z K_ZZ^{-1/2}.

So the whole posterior is a difference of two low-rank pieces plus a diagonal,

    Sigma_* ~= B S B' + sigma^2 I,   B = [s Phi | s^2 Psi],  S = diag(+1..., -1...),

with Psi = K_*n A^{-1/2}. Its top eigenpairs come from a thin QR of B and an
eigendecomposition of the small (r_z + n) square matrix R S R' -- no N x N
matrix is built at any point. Finally D is set from the EXACT marginal
variances, which cost O(N n^2) and need no approximation at all, so the factor
model reproduces every candidate's posterior variance exactly and only the
off-diagonal dependence is approximated. That is the same contract the
eigen-oracle offers, which is what makes the two comparable.

Costs at the full QM9 library (N = 133,702, d = 2048, n = 100, r_z = 256):
every step is one dense matmul or a thin QR; nothing is quadratic in N.

test_factorgp.py checks the kernel algebra against gpytorch's own
mean_cov_from_gp to machine precision before any of it is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------

def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto kernel between count fingerprints, as a single matmul."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dot = A @ B.T
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return dot / (a2 + b2 - dot)


def tanimoto_blocked(A, B, block: int = 8192):
    """Same, in row blocks, for libraries that should not be materialised at once."""
    out = np.empty((len(A), len(B)))
    for a in range(0, len(A), block):
        out[a:a + block] = tanimoto(A[a:a + block], B)
    return out


# --------------------------------------------------------------------------
# exact pieces
# --------------------------------------------------------------------------

class TanimotoPosterior:
    """The exact Tanimoto-GP predictive, in numpy, at O(N n^2) per candidate set."""

    def __init__(self, X_train, y_train, outputscale, noise, mean_constant):
        self.X = np.asarray(X_train, dtype=float)
        self.y = np.asarray(y_train, dtype=float)
        self.s2 = float(outputscale)
        self.sn2 = float(noise)
        self.m0 = float(mean_constant)
        K = tanimoto(self.X, self.X)
        self.A = self.s2 * K + self.sn2 * np.eye(len(K))
        self.L = np.linalg.cholesky(self.A)
        self.alpha = np.linalg.solve(self.A, self.y - self.m0)

    def mean(self, Xs, block: int = 8192) -> np.ndarray:
        out = np.empty(len(Xs))
        for a in range(0, len(Xs), block):
            Kc = tanimoto(Xs[a:a + block], self.X)
            out[a:a + block] = self.m0 + self.s2 * (Kc @ self.alpha)
        return out

    def variance(self, Xs, block: int = 8192, include_noise: bool = True):
        """Exact marginal posterior variance. The Tanimoto kernel has unit
        diagonal, so the prior term is just the outputscale."""
        out = np.empty(len(Xs))
        for a in range(0, len(Xs), block):
            Kc = tanimoto(Xs[a:a + block], self.X)            # (b, n)
            W = np.linalg.solve(self.L, (self.s2 * Kc).T)     # (n, b)
            out[a:a + block] = self.s2 - np.sum(W * W, axis=0)
        if include_noise:
            out = out + self.sn2
        return np.maximum(out, 0.0) + (0.0 if include_noise else 0.0)

    def mean_and_variance(self, Xs, block: int = 8192,
                          include_noise: bool = True):
        """Both, from one pass over the candidate-training kernel.

        Calling mean() and variance() separately recomputes tanimoto(Xs, X)
        twice, and that matmul -- N x 2048 x n -- is the dominant cost of a
        closed-loop round at full library scale.
        """
        n = len(Xs)
        mu = np.empty(n)
        var = np.empty(n)
        for a in range(0, n, block):
            Kc = tanimoto(Xs[a:a + block], self.X)              # (b, n_train)
            mu[a:a + block] = self.m0 + self.s2 * (Kc @ self.alpha)
            W = np.linalg.solve(self.L, (self.s2 * Kc).T)       # (n_train, b)
            var[a:a + block] = self.s2 - np.sum(W * W, axis=0)
        if include_noise:
            var = var + self.sn2
        return mu, np.maximum(var, 0.0)

    def covariance(self, Xs, include_noise: bool = True,
                   block: int = 2048) -> np.ndarray:
        """Dense N x N, filled in row blocks.

        This is the object the released qPO implementation has to build, and
        the reason it prefilters to 10,000 candidates. Written blockwise
        because the naive expression allocates three N x N temporaries and at
        N = 10,000 that is 2.4 GB of churn per acquisition round.
        """
        Xs = np.asarray(Xs)
        N = len(Xs)
        C = np.empty((N, N))
        Wt = np.linalg.solve(self.L, (self.s2 * tanimoto(Xs, self.X)).T).T  # (N, n)
        sq = np.sum(np.asarray(Xs, dtype=np.float64) ** 2, axis=1)
        for a in range(0, N, block):
            Xa = np.asarray(Xs[a:a + block], dtype=np.float64)
            dot = Xa @ np.asarray(Xs, dtype=np.float64).T
            np.divide(dot, sq[a:a + block, None] + sq[None, :] - dot, out=dot)
            dot *= self.s2
            dot -= Wt[a:a + block] @ Wt.T
            C[a:a + block] = dot
        if include_noise:
            C.flat[::N + 1] += self.sn2
        return C


# --------------------------------------------------------------------------
# the factor model, without Sigma
# --------------------------------------------------------------------------

def factor_posterior(post: TanimotoPosterior, Xs, rank: int,
                     inducing: int = 256, seed: int = 0,
                     block: int = 8192, floor_rel: float = 1e-6,
                     inducing_idx=None, return_info: bool = False):
    """Rank-r factor model of the posterior over Xs, in time linear in N.

    Returns (mu, V, D) with V of shape (N, rank) and D the exact marginal
    variances minus the factor part, floored. Nothing of size N x N is formed.
    """
    # Xs stays in whatever dtype it arrives in (float32 for a whole library is
    # 1.1 GB rather than 2.2); each block is widened to float64 inside tanimoto.
    N = len(Xs)
    n = len(post.X)
    rng = np.random.default_rng(seed)
    if inducing_idx is None:
        inducing = min(inducing, N)
        inducing_idx = rng.choice(N, size=inducing, replace=False)
    Z = Xs[np.asarray(inducing_idx)]

    # Phi = K_*Z K_ZZ^{-1/2}   (Nystrom factor of the prior block)
    Kzz = tanimoto(Z, Z)
    w, U = np.linalg.eigh(Kzz)
    keep = w > 1e-10 * max(w.max(), 1e-300)
    Kzz_isqrt = U[:, keep] / np.sqrt(w[keep])            # (rz, k)

    m = Kzz_isqrt.shape[1] + n
    B = np.empty((N, m))
    s = np.sqrt(post.s2)
    for a in range(0, N, block):
        Xa = Xs[a:a + block]
        B[a:a + block, :Kzz_isqrt.shape[1]] = s * (tanimoto(Xa, Z) @ Kzz_isqrt)
        Kc = tanimoto(Xa, post.X)
        B[a:a + block, Kzz_isqrt.shape[1]:] = post.s2 * np.linalg.solve(
            post.L, Kc.T).T
    sign = np.concatenate([np.ones(Kzz_isqrt.shape[1]), -np.ones(n)])

    # top eigenpairs of B diag(sign) B' through a thin QR
    Q, R = np.linalg.qr(B, mode="reduced")
    G = (R * sign[None, :]) @ R.T
    G = 0.5 * (G + G.T)
    lam, W = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1][:rank]
    lam_k = np.maximum(lam[order], 0.0)
    V = (Q @ W[:, order]) * np.sqrt(lam_k)

    mu = post.mean(Xs, block=block)
    var = post.variance(Xs, block=block, include_noise=True)
    d = var - np.sum(V ** 2, axis=1)
    n_floored = int(np.sum(d < floor_rel * np.median(var)))
    d = np.maximum(d, floor_rel * float(np.median(var)))
    if return_info:
        return mu, V, d, {"inducing": len(Z), "kept_inducing_modes": int(keep.sum()),
                          "n_floored": n_floored,
                          "top_eigenvalues": lam[order][:8].tolist()}
    return mu, V, d


def _stream_blocks(post, Xs, Z, Kzz_isqrt, block):
    """Yield (slice, Phi_block, Psi_block) without ever holding all of either.

    Phi = K_*Z K_ZZ^{-1/2} is the Nystrom factor of the prior block; Psi is the
    data correction s^2 K_*n L^{-T}. Both are recomputed per pass rather than
    stored, which is the whole point: at 133,702 candidates with a 2,100-point
    training set the stored version is gigabytes, and each recomputation is one
    matmul.
    """
    for a in range(0, len(Xs), block):
        Xa = Xs[a:a + block]
        Phi = tanimoto(Xa, Z) @ Kzz_isqrt
        Psi = post.s2 * np.linalg.solve(post.L, tanimoto(Xa, post.X).T).T
        yield slice(a, a + len(Xa)), Phi, Psi


def _apply(post, Xs, Z, Kzz_isqrt, block, Omega):
    """M @ Omega for M = s^2 Phi Phi' - Psi Psi', streamed."""
    k = Omega.shape[1]
    rz = Kzz_isqrt.shape[1]
    accA = np.zeros((rz, k))
    accB = np.zeros((len(post.X), k))
    for sl, Phi, Psi in _stream_blocks(post, Xs, Z, Kzz_isqrt, block):
        accA += Phi.T @ Omega[sl]
        accB += Psi.T @ Omega[sl]
    out = np.empty((len(Xs), k))
    for sl, Phi, Psi in _stream_blocks(post, Xs, Z, Kzz_isqrt, block):
        out[sl] = post.s2 * (Phi @ accA) - (Psi @ accB)
    return out


def factor_posterior_streaming(post: TanimotoPosterior, Xs, rank: int,
                               inducing: int = 512, seed: int = 0,
                               block: int = 8192, floor_rel: float = 1e-6,
                               inducing_idx=None, tol: float = 1e-12,
                               return_info: bool = False):
    """The production route: exact top-r eigenpairs in two streaming passes.

    Write the approximated posterior as M = B S B' with B = [s Phi | Psi] and
    S = diag(+1..., -1...). B is N x m with m = r_z + n, which at a full
    library and a grown training set is gigabytes -- so B is never stored.

    Pass one accumulates only the small Gram matrix G = B'B (m x m). Then, with
    G = U diag(g) U' on the retained directions, Q = B U diag(g^-1/2) is an
    orthonormal basis of range(B), and

        Q' M Q = diag(g^-1/2) U' G S G U diag(g^-1/2)

    is an m x m matrix built entirely from G -- no second look at the data.
    Eigendecomposing it gives the eigenvalues of M exactly, because M's range
    is contained in B's. Pass two applies the resulting m x rank matrix to B to
    form V. Memory is O(m^2 + N rank); nothing is randomised and nothing is
    approximated beyond the Nystrom step itself.
    """
    N = len(Xs)
    n = len(post.X)
    rng = np.random.default_rng(seed)
    if inducing_idx is None:
        inducing_idx = rng.choice(N, size=min(inducing, N), replace=False)
    Z = Xs[np.asarray(inducing_idx)]
    Kzz = tanimoto(Z, Z)
    w, U0 = np.linalg.eigh(Kzz)
    keep = w > 1e-10 * max(w.max(), 1e-300)
    Kzz_isqrt = U0[:, keep] / np.sqrt(w[keep])
    rz = Kzz_isqrt.shape[1]
    m = rz + n
    s = np.sqrt(post.s2)
    sign = np.concatenate([np.ones(rz), -np.ones(n)])

    G = np.zeros((m, m))
    for sl, Phi, Psi in _stream_blocks(post, Xs, Z, Kzz_isqrt, block):
        Bb = np.concatenate([s * Phi, Psi], axis=1)
        G += Bb.T @ Bb
    G = 0.5 * (G + G.T)

    g, U = np.linalg.eigh(G)
    k = g > tol * max(g.max(), 1e-300)
    U, g = U[:, k], g[k]
    Gi = U / np.sqrt(g)                          # m x k, = U diag(g^-1/2)
    T = Gi.T @ (G * sign[None, :]) @ G @ Gi      # k x k, equals Q' M Q
    T = 0.5 * (T + T.T)
    lam, W = np.linalg.eigh(T)
    order = np.argsort(lam)[::-1][:rank]
    lam_k = np.maximum(lam[order], 0.0)
    C = Gi @ W[:, order] * np.sqrt(lam_k)        # m x rank, so that V = B C

    V = np.empty((N, len(order)))
    for sl, Phi, Psi in _stream_blocks(post, Xs, Z, Kzz_isqrt, block):
        V[sl] = (s * Phi) @ C[:rz] + Psi @ C[rz:]

    mu = post.mean(Xs, block=block)
    var = post.variance(Xs, block=block, include_noise=True)
    d = var - np.sum(V ** 2, axis=1)
    n_floored = int(np.sum(d < floor_rel * np.median(var)))
    d = np.maximum(d, floor_rel * float(np.median(var)))
    if return_info:
        return mu, V, d, {"inducing": len(Z), "m": m, "kept_modes": int(k.sum()),
                          "n_floored": n_floored,
                          "top_eigenvalues": lam[order][:8].tolist()}
    return mu, V, d


def factor_posterior_randomized(post: TanimotoPosterior, Xs, rank: int,
                                inducing: int = 512, seed: int = 0,
                                block: int = 8192, oversample: int = 10,
                                power_iters: int = 2, floor_rel: float = 1e-6,
                                inducing_idx=None, return_info: bool = False):
    """Same factor model, memory O(N (rank + p)) instead of O(N (r_z + n)).

    A randomised range finder on M = s^2 Phi Phi' - Psi Psi': sketch, a couple
    of subspace iterations to sharpen the leading directions, then a small
    Rayleigh-Ritz. Nothing of size N x (r_z + n) is materialised, so the cost
    of a growing training set is a slightly longer streaming matmul rather than
    gigabytes of storage. Checked against the exact QR route in
    test_factorgp.py.
    """
    N = len(Xs)
    rng = np.random.default_rng(seed)
    if inducing_idx is None:
        inducing_idx = rng.choice(N, size=min(inducing, N), replace=False)
    Z = Xs[np.asarray(inducing_idx)]
    Kzz = tanimoto(Z, Z)
    w, U = np.linalg.eigh(Kzz)
    keep = w > 1e-10 * max(w.max(), 1e-300)
    Kzz_isqrt = U[:, keep] / np.sqrt(w[keep])

    k = min(rank + oversample, N)
    Q = rng.standard_normal((N, k))
    for _ in range(power_iters + 1):
        Q = _apply(post, Xs, Z, Kzz_isqrt, block, Q)
        Q, _ = np.linalg.qr(Q, mode="reduced")
    T = Q.T @ _apply(post, Xs, Z, Kzz_isqrt, block, Q)
    T = 0.5 * (T + T.T)
    lam, W = np.linalg.eigh(T)
    order = np.argsort(lam)[::-1][:rank]
    lam_k = np.maximum(lam[order], 0.0)
    V = (Q @ W[:, order]) * np.sqrt(lam_k)

    mu = post.mean(Xs, block=block)
    var = post.variance(Xs, block=block, include_noise=True)
    d = var - np.sum(V ** 2, axis=1)
    n_floored = int(np.sum(d < floor_rel * np.median(var)))
    d = np.maximum(d, floor_rel * float(np.median(var)))
    if return_info:
        return mu, V, d, {"inducing": len(Z), "n_floored": n_floored,
                          "top_eigenvalues": lam[order][:8].tolist()}
    return mu, V, d


def load_gp(snapshot_dir: Path):
    """Rebuild the posterior object from what snapshot.py saved."""
    snapshot_dir = Path(snapshot_dir)
    meta = json.loads((snapshot_dir / "meta.json").read_text())
    hp = meta["gp_hyperparameters"]
    X = np.load(snapshot_dir / "gp_train_x.npy").astype(float)
    y = np.load(snapshot_dir / "gp_train_y.npy").astype(float)
    return TanimotoPosterior(X, y, hp["covar_module.outputscale"],
                             hp["likelihood.noise"],
                             hp["mean_module.constant"]), meta
