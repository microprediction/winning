"""`Tree.from_linkage` promises the cophenetic matrix, or says no.

Its docstring states the premise -- "increments nonnegative by linkage
monotonicity" -- and the code never checked it. `centroid` and `median`
linkage routinely invert, and `max(lam2, 0.0)` then clipped the negative
increment away, returning a tree whose implied covariance is NOT the
cophenetic one it promises (#133).

A tree race is a nested variance decomposition. An inversion -- a merge
below its own parent -- has no representation in one, so this refuses
rather than answering a different question.
"""
import numpy as np
import pytest

from scipy.cluster.hierarchy import cophenet, is_monotonic, linkage
from scipy.spatial.distance import squareform

from winning.factor.structures import Tree

HORIZON = 1.0 / np.sqrt(2.0)


def _implied_cov(t, n):
    parent = np.asarray(t.parent)
    anc = []
    for i in range(n):
        a, u = set(), i
        while parent[u] >= 0:
            a.add(parent[u])
            u = parent[u]
        anc.append(a)
    S = np.diag(np.asarray(t.D, float))
    q = np.asarray(t.strength, float) ** 2
    for i in range(n):
        for j in range(n):
            S[i, j] += sum(q[k] for k in anc[i] & anc[j])
    return S


def _cophenetic(Z):
    C = 1 - 2 * squareform(cophenet(Z)) ** 2
    np.fill_diagonal(C, 1)
    return C


def test_a_non_monotonic_linkage_is_refused():
    X = 0.2 * np.random.default_rng(1).normal(size=(8, 3))
    Z = linkage(X, method="centroid")
    assert not is_monotonic(Z), "fixture is no longer a hard case"
    with pytest.raises(ValueError, match="not monotonic"):
        Tree.from_linkage(Z)


@pytest.mark.parametrize("method", ["average", "complete", "ward", "single"])
def test_monotonic_methods_are_unaffected(method):
    X = 0.2 * np.random.default_rng(1).normal(size=(8, 3))
    Z = linkage(X, method=method)
    t = Tree.from_linkage(Z)
    assert len(t.parent) == 15


def test_the_refusal_tracks_scipy_exactly():
    """The criterion is on rho increments, scipy's is on raw heights, so
    they could in principle disagree. Over 240 linkages they do not, and
    nothing monotonic is refused."""
    refused_wrongly, missed = 0, 0
    for seed in range(60):
        for method in ("centroid", "median", "average", "ward"):
            X = 0.2 * np.random.default_rng(seed).normal(size=(8, 3))
            Z = linkage(X, method=method)
            mono = is_monotonic(Z)
            try:
                Tree.from_linkage(Z)
                refused = False
            except ValueError:
                refused = True
            refused_wrongly += refused and mono
            missed += (not refused) and (not mono)
    assert refused_wrongly == 0, f"{refused_wrongly} monotonic linkages refused"
    assert missed == 0, f"{missed} inversions accepted"


def test_what_is_accepted_is_exact():
    """The promise, on the cases where the documented negative-correlation
    floor does not apply: merges above h = 1/sqrt(2) are deliberately
    floored to zero correlation and are excluded here for that reason."""
    worst, n_cases = 0.0, 0
    for seed in range(60):
        for method in ("centroid", "median", "average", "ward"):
            X = 0.2 * np.random.default_rng(seed).normal(size=(8, 3))
            Z = linkage(X, method=method)
            if (Z[:, 2] > HORIZON).any():
                continue
            try:
                t = Tree.from_linkage(Z)
            except ValueError:
                continue
            worst = max(worst, np.max(np.abs(_implied_cov(t, 8)
                                             - _cophenetic(Z))))
            n_cases += 1
    assert n_cases > 50, f"only {n_cases} cases exercised"
    assert worst < 1e-12, f"implied covariance is not cophenetic: {worst:.2e}"


def test_the_message_says_how_far_and_how_many():
    """'not monotonic' alone leaves the caller to find the inversion."""
    X = 0.2 * np.random.default_rng(1).normal(size=(8, 3))
    with pytest.raises(ValueError) as e:
        Tree.from_linkage(linkage(X, method="centroid"))
    msg = str(e.value)
    assert "BELOW its parent" in msg
    assert "node(s) do" in msg
    assert "is_monotonic" in msg
