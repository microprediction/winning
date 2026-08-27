"""Anchors for the Sigma-free factor GP.

The numpy posterior in factorgp.py re-derives what gpytorch computes. If that
re-derivation is off by a hyperparameter convention -- an outputscale applied
once instead of twice, noise added to the wrong term -- every downstream
number is wrong in a way no self-consistency check would catch. So it is
checked directly against the authors' own mean_cov_from_gp on a real snapshot,
to machine precision, before anything else here is used.

Then the Nystrom factor model is checked to do what it claims: exact marginal
variances, and probabilities close to those of the eigen-oracle at equal rank.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from factorgp import (TanimotoPosterior, factor_posterior, load_gp,  # noqa: E402
                      tanimoto)
from factorize import eig_factor  # noqa: E402
from metrics import tv_error  # noqa: E402
from pom import pom_fast, sobol_nodes  # noqa: E402

SNAP = HERE.parent / "snapshots" / "qm9_gap_seed7"
need_snapshot = pytest.mark.skipif(
    not (SNAP / "meta.json").exists() or not (SNAP / "gp_train_x.npy").exists(),
    reason="snapshot not built")


def test_tanimoto_matches_the_authors_kernel():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, os.path.expanduser("~/github/qPO"))
    from gp.tanimoto_gp import batch_tanimoto_sim

    rng = np.random.default_rng(0)
    A = rng.integers(0, 5, (17, 40)).astype(float)
    B = rng.integers(0, 5, (11, 40)).astype(float)
    ours = tanimoto(A, B)
    theirs = batch_tanimoto_sim(torch.as_tensor(A), torch.as_tensor(B)).numpy()
    assert np.max(np.abs(ours - theirs)) < 1e-12
    assert np.max(np.abs(np.diag(tanimoto(A, A)) - 1.0)) < 1e-12


@need_snapshot
def test_numpy_posterior_matches_gpytorch():
    """The decisive anchor: same mu and same Sigma as the authors' code path."""
    post, meta = load_gp(SNAP)
    N = 500
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(float)
    mu_ref = np.load(SNAP / f"N{N}" / "mu.npy")
    S_ref = np.load(SNAP / f"N{N}" / "Sigma.npy")
    c = meta.get("c", 1)

    mu = c * post.mean(Xs)
    S = post.covariance(Xs, include_noise=True)
    v = post.variance(Xs, include_noise=True)

    assert np.max(np.abs(mu - mu_ref)) < 1e-9, np.max(np.abs(mu - mu_ref))
    assert np.max(np.abs(S - S_ref)) < 1e-11, np.max(np.abs(S - S_ref))
    assert np.max(np.abs(v - np.diag(S_ref))) < 1e-11


@need_snapshot
def test_nystrom_factor_model_has_exact_marginal_variances():
    post, meta = load_gp(SNAP)
    N = 1000
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(float)
    S_ref = np.load(SNAP / f"N{N}" / "Sigma.npy")
    mu, V, d = factor_posterior(post, Xs, rank=8, inducing=256, seed=0)
    recon_diag = np.sum(V ** 2, axis=1) + d
    # exact wherever the floor did not bite
    err = np.abs(recon_diag - np.diag(S_ref))
    assert np.median(err) < 1e-12
    assert np.mean(err < 1e-10) > 0.99, np.mean(err < 1e-10)


@need_snapshot
@pytest.mark.parametrize("rank", [2, 8])
def test_nystrom_probabilities_track_the_eigen_oracle(rank):
    """The sparse route must land in the same place as the N x N oracle."""
    post, meta = load_gp(SNAP)
    N = 1000
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(float)
    S_ref = np.load(SNAP / f"N{N}" / "Sigma.npy")
    mu_ref = np.load(SNAP / f"N{N}" / "mu.npy")
    c = meta.get("c", 1)

    F, W = sobol_nodes(rank, m=9, seed=0)
    Vo, do = eig_factor(S_ref, rank)
    p_oracle = pom_fast(mu_ref, Vo, do, F, W, points=129)

    mu_n, Vn, dn = factor_posterior(post, Xs, rank=rank, inducing=512, seed=0)
    p_nys = pom_fast(c * mu_n, Vn, dn, F, W, points=129)

    tv = tv_error(p_oracle, p_nys)
    assert tv < 0.05, tv


@need_snapshot
def test_nystrom_improves_with_more_inducing_points():
    """More inducing molecules must move it toward the oracle, not away."""
    post, meta = load_gp(SNAP)
    N = 1000
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(float)
    S_ref = np.load(SNAP / f"N{N}" / "Sigma.npy")
    mu_ref = np.load(SNAP / f"N{N}" / "mu.npy")
    c = meta.get("c", 1)
    rank = 4
    F, W = sobol_nodes(rank, m=9, seed=0)
    Vo, do = eig_factor(S_ref, rank)
    p_oracle = pom_fast(mu_ref, Vo, do, F, W, points=129)

    errs = []
    for m in (64, 256, 1000):
        mu_n, Vn, dn = factor_posterior(post, Xs, rank=rank, inducing=m, seed=0)
        errs.append(tv_error(p_oracle, pom_fast(c * mu_n, Vn, dn, F, W, points=129)))
    assert errs[-1] < errs[0], errs
    # with every candidate an inducing point the Nystrom step is exact
    assert errs[-1] < 5e-3, errs


@need_snapshot
def test_randomized_matches_the_exact_qr_route():
    """The streaming range finder must land on the same factor model."""
    from factorgp import factor_posterior_randomized
    post, meta = load_gp(SNAP)
    N = 1000
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(np.float32)
    mu_ref = np.load(SNAP / f"N{N}" / "mu.npy")
    c = meta.get("c", 1)
    rank = 4
    idx = np.arange(0, N, 2)          # same inducing set for both routes
    F, W = sobol_nodes(rank, m=9, seed=0)

    mu_a, Va, da = factor_posterior(post, Xs, rank=rank, inducing_idx=idx)
    mu_b, Vb, db = factor_posterior_randomized(post, Xs, rank=rank,
                                               inducing_idx=idx, seed=1,
                                               power_iters=3)
    assert np.max(np.abs(mu_a - mu_b)) < 1e-12
    # V is fixed only up to sign and rotation, so compare the spectrum. The
    # trailing retained eigenvalue is the one a range finder resolves worst;
    # with 10 oversamples it lands within about 1e-3 relative, and more power
    # iterations tighten it. What has to be tight is the probabilities.
    la = np.sort(np.sum(Va ** 2, 0))
    lb = np.sort(np.sum(Vb ** 2, 0))
    assert np.max(np.abs(la - lb) / la) < 5e-3, (la, lb)
    pa = pom_fast(c * mu_a, Va, da, F, W, points=129)
    pb = pom_fast(c * mu_b, Vb, db, F, W, points=129)
    assert tv_error(pa, pb) < 2e-3, tv_error(pa, pb)

    # and it converges: more oversampling brings the spectrum in
    _, Vc, _ = factor_posterior_randomized(post, Xs, rank=rank, inducing_idx=idx,
                                           seed=1, power_iters=3, oversample=60)
    lc = np.sort(np.sum(Vc ** 2, 0))
    assert np.max(np.abs(lc - la) / la) < np.max(np.abs(lb - la) / la)


@need_snapshot
@pytest.mark.parametrize("rank", [2, 8])
def test_streaming_route_is_exact_against_the_qr_route(rank):
    """Two streaming passes must reproduce the dense QR answer, not approximate it."""
    from factorgp import factor_posterior_streaming
    post, meta = load_gp(SNAP)
    N = 1000
    Xs = np.load(SNAP / f"N{N}" / "fps.npy").astype(np.float32)
    c = meta.get("c", 1)
    idx = np.arange(0, N, 3)
    F, W = sobol_nodes(rank, m=9, seed=0)

    mu_a, Va, da = factor_posterior(post, Xs, rank=rank, inducing_idx=idx)
    mu_b, Vb, db = factor_posterior_streaming(post, Xs, rank=rank,
                                              inducing_idx=idx, block=256)
    la = np.sort(np.sum(Va ** 2, 0))
    lb = np.sort(np.sum(Vb ** 2, 0))
    assert np.max(np.abs(la - lb) / la) < 1e-8, (la, lb)
    assert np.max(np.abs(da - db)) < 1e-10 * np.median(da)
    # The claim is about the MODEL, so compare V V' + D. Comparing probabilities
    # instead would fail at 1e-4 for a reason that has nothing to do with the
    # factorisation: V is fixed only up to the sign of each column, V V' is not
    # affected, but a fixed Sobol node set is not sign-symmetric, so the two
    # answers differ by the quadrature's own error.
    Sa = Va @ Va.T + np.diag(da)
    Sb = Vb @ Vb.T + np.diag(db)
    assert np.max(np.abs(Sa - Sb)) < 1e-12 * np.median(da), np.max(np.abs(Sa - Sb))
    # The two must also agree on probabilities to within what the quadrature
    # itself can resolve at this budget -- measured here as the spread between
    # two independent scrambles, not assumed.
    Fh, Wh = sobol_nodes(rank, m=13, seed=0)
    Fh2, Wh2 = sobol_nodes(rank, m=13, seed=77)
    pa = pom_fast(c * mu_a, Va, da, Fh, Wh, points=129)
    pb = pom_fast(c * mu_b, Vb, db, Fh, Wh, points=129)
    self_tv = tv_error(pa, pom_fast(c * mu_a, Va, da, Fh2, Wh2, points=129))
    assert tv_error(pa, pb) < 3 * self_tv + 1e-9, (tv_error(pa, pb), self_tv)


@need_snapshot
def test_fused_mean_and_variance_matches_the_separate_calls():
    post, meta = load_gp(SNAP)
    Xs = np.load(SNAP / "N1000" / "fps.npy").astype(np.float32)
    mu, var = post.mean_and_variance(Xs)
    assert np.max(np.abs(mu - post.mean(Xs))) < 1e-14
    assert np.max(np.abs(var - post.variance(Xs))) < 1e-14
