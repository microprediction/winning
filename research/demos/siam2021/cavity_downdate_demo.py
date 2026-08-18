"""Rank-one cavity identity: one inverse contains every leave-one-out inverse.

For SPD A with G = A^{-1}, the inverse of the principal submatrix A_{-i,-i} is

    G^(i)_{jk} = G_{jk} - G_{ji} G_{ik} / G_{ii},   j, k != i

so after one O(n^3) factorization, all n single-deletion ("cavity") inverses are
rank-one downdates, and any scalar leave-one-out observable is O(1). The subtracted
term is the interaction between j and k mediated through i.

Also verifies leave-k-out:  G^(S) = G_{S̄,S̄} - G_{S̄,S} G_{S,S}^{-1} G_{S,S̄}.

Requires only numpy. Run:  python research/demos/cavity_downdate_demo.py
"""

import time

import numpy as np


def random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned random SPD matrix (e.g. a stiffness/precision matrix)."""
    M = rng.standard_normal((n, n))
    return M @ M.T / n + np.eye(n)


def cavity_downdate(G: np.ndarray, i: int) -> np.ndarray:
    """Leave-one-out inverse via the rank-one downdate of the full inverse."""
    keep = np.arange(G.shape[0]) != i
    g_col = G[keep, i]
    return G[np.ix_(keep, keep)] - np.outer(g_col, g_col) / G[i, i]


def cavity_downdate_block(G: np.ndarray, S: list[int]) -> np.ndarray:
    """Leave-k-out inverse via a k x k solve on the full inverse."""
    keep = np.setdiff1d(np.arange(G.shape[0]), S)
    G_kS = G[np.ix_(keep, S)]
    return G[np.ix_(keep, keep)] - G_kS @ np.linalg.solve(G[np.ix_(S, S)], G_kS.T)


def main() -> None:
    rng = np.random.default_rng(7)

    # --- correctness, small n ------------------------------------------------
    n = 40
    A = random_spd(n, rng)
    G = np.linalg.inv(A)

    worst = 0.0
    for i in range(n):
        keep = np.arange(n) != i
        direct = np.linalg.inv(A[np.ix_(keep, keep)])
        worst = max(worst, np.abs(cavity_downdate(G, i) - direct).max())
    print(f"leave-1-out: max |downdate - direct inverse| over all {n} deletions: {worst:.3e}")

    for k in (2, 3):
        S = list(rng.choice(n, size=k, replace=False))
        keep = np.setdiff1d(np.arange(n), S)
        direct = np.linalg.inv(A[np.ix_(keep, keep)])
        err = np.abs(cavity_downdate_block(G, S) - direct).max()
        print(f"leave-{k}-out: max error for S={S}: {err:.3e}")

    # --- timing, larger n ----------------------------------------------------
    n = 500
    A = random_spd(n, rng)

    t0 = time.perf_counter()
    for i in range(n):
        keep = np.arange(n) != i
        np.linalg.inv(A[np.ix_(keep, keep)])
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = np.linalg.inv(A)
    for i in range(n):
        cavity_downdate(G, i)
    t_fast = time.perf_counter() - t0

    print(f"\nn={n}: all {n} leave-one-out inverses")
    print(f"  naive   ({n} fresh inversions): {t_naive:.2f}s")
    print(f"  cavity  (1 inversion + downdates): {t_fast:.2f}s   ({t_naive / t_fast:.0f}x)")

    # Scalar observables are O(1) each: e.g. every leave-one-out log-det via
    # log det A_{-i,-i} = log det A + log G_ii  (Schur determinant identity).
    logdet = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(A))))
    loo_logdets = logdet + np.log(np.diag(G))
    i = 123
    keep = np.arange(n) != i
    direct = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(A[np.ix_(keep, keep)]))))
    print(f"\nscalar example, log det A_(-i,-i): downdate {loo_logdets[i]:.10f} "
          f"vs direct {direct:.10f}")


if __name__ == "__main__":
    main()
