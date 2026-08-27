"""What counts as agreement between two qPO acquisition scores.

Probability error, rank correlation, batch overlap, and -- the one that decides
whether a difference matters operationally -- the qPO objective efficiency

    eta = sum_{i in S_fast} p_i^ref  /  sum_{i in S_ref} p_i^ref,

the value of the batch the fast method picks, scored under the reference
probabilities, against the best batch those reference probabilities allow. The
qPO batch objective is additive, so this is exactly the fraction of the
achievable objective the fast method keeps. Two methods can disagree about
which molecules to buy and still both be right, if the molecules they disagree
over carry the same probability.
"""

from __future__ import annotations

import numpy as np


def tv_error(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def l1_error(p, q):
    return float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def max_abs_error(p, q):
    return float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float))))


def spearman(p, q):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(p, float), np.asarray(q, float)).statistic
    return float(r)


def kendall_top(p, q, k: int = 100):
    """Kendall tau restricted to the union of the two top-k sets."""
    from scipy.stats import kendalltau
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    idx = np.union1d(np.argsort(-p)[:k], np.argsort(-q)[:k])
    return float(kendalltau(p[idx], q[idx]).statistic)


def select_batch(p, b: int, mu=None):
    """The authors' selection rule, indices of the chosen batch.

    acquire_qPO sorts on the pair (-probability, -mean), so ties in the
    probability are broken by the posterior mean. That detail is not cosmetic:
    with the shipped 10,000 samples and probabilities near 1/N, most candidates
    share a winner count, and the mean is doing much of the selecting. Scoring
    their estimator with index-order tie-breaking would be a straw man, so the
    same rule is applied to every method here. It changes nothing for methods
    whose scores are continuous.
    """
    p = np.asarray(p, dtype=float)
    if mu is None:
        return np.argsort(-p, kind="stable")[:b]
    order = np.lexsort((-np.asarray(mu, dtype=float), -p))
    return order[:b]


def top_set(p, b: int, mu=None):
    return set(select_batch(p, b, mu).tolist())


def batch_agreement(p_ref, p_test, sizes=(10, 25, 50, 100), mu=None) -> dict:
    out = {}
    for b in sizes:
        A = top_set(p_ref, b, mu)
        B = top_set(p_test, b, mu)
        inter = len(A & B)
        out[f"top{b}_recall"] = inter / b
        out[f"top{b}_jaccard"] = inter / len(A | B)
    return out


def qpo_efficiency(p_ref, p_test, b: int = 100, mu=None) -> float:
    """Value of the test batch under the reference probabilities, over the best."""
    p_ref = np.asarray(p_ref, dtype=float)
    S_test = select_batch(p_test, b, mu)
    S_ref = select_batch(p_ref, b, mu)
    denom = float(p_ref[S_ref].sum())
    if denom <= 0:
        return float("nan")
    return float(p_ref[S_test].sum() / denom)


def oracle_batch_value(oracle, p_test, b: int = 100, mu=None) -> dict:
    """What the batch is actually worth, using the held-out true objective."""
    oracle = np.asarray(oracle, dtype=float)
    S = select_batch(p_test, b, mu)
    return {"batch_oracle_mean": float(oracle[S].mean()),
            "batch_oracle_max": float(oracle[S].max()),
            "batch_oracle_top10": float(np.sort(oracle[S])[-10:].mean())}


def compare(p_ref, p_test, b: int = 100, sizes=(10, 25, 50, 100), mu=None) -> dict:
    out = {
        "tv_error": tv_error(p_ref, p_test),
        "l1_error": l1_error(p_ref, p_test),
        "max_abs_error": max_abs_error(p_ref, p_test),
        "spearman": spearman(p_ref, p_test),
        "qpo_efficiency": qpo_efficiency(p_ref, p_test, b, mu),
    }
    out.update(batch_agreement(p_ref, p_test, sizes, mu))
    return out


# --------------------------------------------------------------------------
# batch diversity
# --------------------------------------------------------------------------

def tanimoto_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto (Jaccard on counts) for count fingerprints, vectorised."""
    X = np.asarray(X, dtype=float)
    dot = X @ X.T
    sq = np.sum(X ** 2, axis=1)
    denom = sq[:, None] + sq[None, :] - dot
    return dot / np.maximum(denom, 1e-300)


def diversity_stats(fps: np.ndarray, idx, threshold: float = 0.4) -> dict:
    idx = np.asarray(list(idx), dtype=int)
    T = tanimoto_matrix(np.asarray(fps, dtype=float)[idx])
    iu = np.triu_indices(len(idx), k=1)
    v = T[iu]
    return {
        "mean_tanimoto": float(v.mean()),
        "median_tanimoto": float(np.median(v)),
        "p90_tanimoto": float(np.percentile(v, 90)),
        "max_tanimoto": float(v.max()),
        f"frac_pairs_gt_{threshold}": float(np.mean(v > threshold)),
    }
