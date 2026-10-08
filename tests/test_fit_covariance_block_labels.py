"""fit_covariance(C, blocks=labels): a caller's block membership replaces
the clustering stage (#621).

The setting: candidates evaluated on shared trials and descended from one
another by edits, so the families are KNOWN. Clustering the residual only
rediscovers them, imperfectly; given labels should fit the family
structure at least as well and leave everything else of the pipeline
alone.
"""
import numpy as np
import pytest

from winning.factor.core import fit_covariance


def _family_cov(seed=3, n_fam=6, size=4, k=3):
    rng = np.random.default_rng(seed)
    n = n_fam * size
    fam = np.repeat(np.arange(n_fam), size)
    G = rng.normal(0, 0.6, (n, k))
    B = np.zeros((n, n_fam))
    B[np.arange(n), fam] = rng.uniform(0.4, 0.8, n)
    C = G @ G.T + B @ B.T + np.diag(rng.uniform(0.2, 0.5, n))
    return C, fam


def test_clustered_labels_given_back_reproduce_the_clustered_fit():
    """Labels replace the clustering and nothing else: handing the fit
    the membership it clustered itself returns the same fit, bit for
    bit."""
    C, _ = _family_cov()
    V1, D1, F1, W1, r1 = fit_covariance(C, k=3, blocks=6,
                                        return_report=True)
    V2, D2, F2, W2, r2 = fit_covariance(C, k=3, blocks=r1["block_labels"],
                                        return_report=True)
    assert r1["blocks"] == "clustered" and r2["blocks"] == "given"
    assert np.array_equal(V1, V2) and np.array_equal(D1, D2)
    assert np.array_equal(F1, F2) and np.array_equal(W1, W2)
    assert np.array_equal(r1["block_labels"], r2["block_labels"])


def test_known_families_fit_better_than_clustering():
    """The issue's complaint: the clustered fit leaves the contrast
    residual above the 0.05 warning line, the true families do not."""
    C, fam = _family_cov()
    *_, r_clu = fit_covariance(C, k=3, return_report=True)
    *_, r_giv = fit_covariance(C, k=3, blocks=fam, return_report=True)
    assert r_giv["blocks"] == "given"
    assert r_clu["contrast_residual_max"] > 0.05
    assert r_giv["contrast_residual_max"] < 0.05
    assert r_giv["projected_residual_rel"] < r_clu["projected_residual_rel"]


def test_label_spelling_does_not_matter():
    """A label names a block; 1-based, gapped, negative or float-integer
    spellings of one partition are one partition."""
    C, fam = _family_cov(seed=5)
    ref = fit_covariance(C, k=3, blocks=fam)
    for lab in (fam + 1, 10 * fam - 7, fam.astype(float), list(fam)):
        out = fit_covariance(C, k=3, blocks=lab)
        for a, b in zip(ref, out):
            assert np.array_equal(a, b)
    # an order-reversing relabelling permutes the block COLUMNS only
    V, D, *_ = fit_covariance(C, k=3, blocks=(5 - fam) * 3)
    assert np.array_equal(D, ref[1])
    assert np.allclose(V @ V.T, ref[0] @ ref[0].T, rtol=0, atol=1e-13)


def test_singleton_blocks_get_no_loading():
    """A singleton block has no off-diagonal residual, so -- as on the
    clustered path -- it carries no block column. All-singleton labels
    are therefore the no-blocks fit."""
    C, _ = _family_cov(seed=7)
    n = len(C)
    none = fit_covariance(C, k=3, blocks=1)
    single = fit_covariance(C, k=3, blocks=np.arange(n))
    for a, b in zip(none, single):
        assert np.array_equal(a, b)
    # mixed: one family plus singletons adds exactly one column at most
    lab = np.arange(n)
    lab[:4] = -1
    V, *_ = fit_covariance(C, k=3, blocks=lab)
    assert V.shape[1] <= none[0].shape[1] + 1


def test_scale_and_permutation():
    """The unit-scale recursion carries the labels, and permuting the
    entrants with their labels permutes the fitted covariance."""
    C, fam = _family_cov(seed=11)
    V, D, *_ = fit_covariance(C, k=3, blocks=fam)
    V4, D4, *_ = fit_covariance(16.0 * C, k=3, blocks=fam)
    assert np.allclose(V4, 4.0 * V, rtol=1e-12, atol=0)
    assert np.allclose(D4, 16.0 * D, rtol=1e-12, atol=0)
    perm = np.random.default_rng(0).permutation(len(C))
    Vp, Dp, *_ = fit_covariance(C[np.ix_(perm, perm)], k=3,
                                blocks=fam[perm])
    S = V @ V.T + np.diag(D)
    Sp = Vp @ Vp.T + np.diag(Dp)
    assert np.allclose(Sp, S[np.ix_(perm, perm)], atol=1e-8)


def test_report_states_the_source():
    C, fam = _family_cov(seed=2)
    assert fit_covariance(C, k=3, return_report=True)[4]["blocks"] \
        == "clustered"
    assert fit_covariance(C, k=3, blocks=1, return_report=True)[4][
        "blocks"] == "none"
    rep = fit_covariance(C, k=3, blocks=fam, return_report=True)[4]
    assert rep["blocks"] == "given"
    assert np.array_equal(rep["block_labels"], fam)
    assert rep["arm"] in ("pipeline", "eigen")
    # a one-runner field reports the same keys
    rep1 = fit_covariance([[2.0]], blocks=[4], return_report=True)[4]
    assert rep1["blocks"] == "given"


@pytest.mark.parametrize("bad, match", [
    (np.zeros(5, dtype=int), "labels for"),            # wrong length
    (np.zeros((24, 1), dtype=int), "1-D"),             # not a vector
    (np.r_[np.zeros(23), 0.5], "whole numbers"),       # fractional
    (np.r_[np.zeros(23), np.nan], "NaN"),              # missing
    (np.r_[np.zeros(23), np.inf], "NaN or inf"),
    (np.array(["a"] * 24), "integers"),                # not numeric
    (np.ones(24, dtype=bool), "integers"),             # a mask, not labels
])
def test_label_validation(bad, match):
    C, _ = _family_cov()
    with pytest.raises(ValueError, match=match):
        fit_covariance(C, k=3, blocks=bad)
