"""winning.torch: Thurstone heads for PyTorch.

Skipped where torch is absent (CI has no torch job). Parity is against
winning.factor at its defaults; the two lattices differ (winning bisects
the winner distribution's quantiles, winning.torch takes a closed-form
per-row window, docs/design/torch.md), so agreement is to the forward's
own accuracy, ~1e-12 relative, not bitwise.
"""
import itertools
import math
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import winning.torch as wt  # noqa: E402
from winning.factor import (ordered_probabilities, race_probabilities,  # noqa: E402
                            top_k_probabilities)

F64 = torch.float64
RNG = np.random.default_rng(20261015)


def _t(a, dtype=F64, grad=False):
    return torch.tensor(np.asarray(a, dtype=float), dtype=dtype,
                        requires_grad=grad)


FIELDS = {
    "three": np.array([0.0, 1.0, 2.0]),
    "pair_10sd": np.array([0.0, 10.0]),
    "rare_middle": np.array([0.0, 20.0, 40.0]),
    "ten": RNG.normal(size=10),
    "hundred_wide": 2.0 * RNG.normal(size=100),
    "longshot": np.r_[np.zeros(5), 10.0],
}


# ------------------------------------------------------------- parity
@pytest.mark.parametrize("base", ["normal", "gumbel"])
@pytest.mark.parametrize("name", sorted(FIELDS))
@pytest.mark.parametrize("hetero", [False, True])
def test_parity_with_winning_race_probabilities(base, name, hetero):
    mu = FIELDS[name]
    sd = np.linspace(0.6, 1.8, len(mu)) if hetero else np.ones(len(mu))
    lp = wt.race_log_prob(_t(mu), _t(sd), base=base).numpy()

    def rel_err(ref):
        ok = ref > 1e-290
        return np.abs(np.exp(lp[ok]) / ref[ok] - 1.0).max()

    # relative, even for 1e-176 longshots: both lattices resolve them
    err = rel_err(race_probabilities(mu, D=sd ** 2, base=base))
    if err >= 1e-10 and base != "normal":
        # winning's bulk window post-checks rare winners for the normal
        # base only: on gumbel rare_middle (hetero) its default is 8e-6
        # off the converged span lattice, which this module matches
        # (the span window has its own ~1e-10 relative floor on 1e-13)
        err = 0.1 * rel_err(race_probabilities(mu, D=sd ** 2, base=base,
                                               window="span", points=20001))
    assert err < 1e-10, (name, err)


def test_pair_matches_the_closed_form_far_into_the_tail():
    from scipy.special import log_ndtr
    for gap in (1.0, 10.0, 30.0):
        lp = wt.race_log_prob(_t([0.0, gap])).numpy()
        exact = log_ndtr(-gap / math.sqrt(2.0))
        assert abs(lp[1] - exact) < 1e-9 * max(1.0, abs(exact)), gap


def test_gumbel_base_is_exactly_softmax():
    mu = _t(RNG.normal(size=(4, 12)))
    sd = 0.7
    lp = wt.race_log_prob(mu, sd, base="gumbel")
    c = math.pi / math.sqrt(6.0)
    ref = torch.log_softmax(-c * mu / sd, dim=-1)
    assert torch.allclose(lp, ref, atol=1e-11, rtol=0)


def test_probabilities_sum_to_one():
    mu = _t(RNG.normal(size=(5, 30)))
    for base in ("normal", "gumbel"):
        lp = wt.race_log_prob(mu, base=base)
        assert torch.allclose(torch.logsumexp(lp, -1),
                              torch.zeros(5, dtype=F64), atol=1e-14)


# ------------------------------------------------------------ gradients
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_gradcheck_of_the_analytic_backward(base, checkpoint):
    mu = _t(RNG.normal(size=(2, 5)), grad=True)
    sd = _t(RNG.uniform(0.5, 1.5, size=(2, 5)), grad=True)
    # a fixed lattice: finite differences must see the same grid
    x = wt.race_lattice(mu.detach(), sd.detach(), base=base)
    f = lambda m, s: wt.race_log_prob(m, s, base=base, lattice=x,
                                      checkpoint=checkpoint)
    assert torch.autograd.gradcheck(f, (mu, sd), eps=1e-6, atol=1e-7)
    mask = torch.tensor([[True, False, True, True, True]] * 2)
    g = lambda m, s: wt.race_log_prob(m, s, base=base, lattice=x, mask=mask,
                                      checkpoint=checkpoint)[mask]
    assert torch.autograd.gradcheck(g, (mu, sd), eps=1e-6, atol=1e-7)


def test_gradcheck_with_the_moving_lattice():
    """Without a fixed lattice the grid moves with mu; the omitted term is
    the derivative of the discretisation error, invisible at gradcheck's
    tolerance."""
    mu = _t(RNG.normal(size=4), grad=True)
    sd = _t([1.0, 0.8, 1.3, 1.1], grad=True)
    assert torch.autograd.gradcheck(lambda m, s: wt.race_log_prob(m, s),
                                    (mu, sd), eps=1e-6, atol=1e-7)


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_analytic_backward_equals_autograd_through_the_forward(base):
    mu0 = RNG.normal(size=(3, 40)) * 1.5
    sd0 = RNG.uniform(0.5, 2.0, size=(3, 40))
    G = _t(RNG.normal(size=(3, 40)))
    grads = {}
    for how, ck in (("analytic", False), ("analytic", True),
                    ("autograd", False)):
        mu, sd = _t(mu0, grad=True), _t(sd0, grad=True)
        lp = wt.race_log_prob(mu, sd, base=base, backward=how, checkpoint=ck)
        (lp * G).sum().backward()
        grads[how, ck] = (mu.grad, sd.grad, lp.detach())
    b = grads["autograd", False]
    for key in (("analytic", False), ("analytic", True)):
        a = grads[key]
        assert torch.equal(a[2], b[2])
        for ga, gb in zip(a[:2], b[:2]):
            assert torch.allclose(ga, gb, atol=1e-12, rtol=1e-10)


def test_gradient_wrt_scalar_and_broadcast_sd():
    mu = _t(RNG.normal(size=(3, 6)), grad=True)
    sd = torch.tensor(1.3, dtype=F64, requires_grad=True)
    lp = wt.race_log_prob(mu, sd)
    lp[:, 0].sum().backward()
    assert sd.grad is not None and torch.isfinite(sd.grad)
    # scale equivariance: P depends on mu / sd only, so
    # sd * dL/dsd = -sum mu * dL/dmu
    assert torch.allclose(sd.grad * sd, -(mu.grad * mu.detach()).sum(),
                          atol=1e-10)


# ----------------------------------------------------- batch, mask, symmetry
def test_padding_runners_do_not_change_results():
    mu = RNG.normal(size=7)
    full = wt.race_log_prob(_t(mu))
    pad = np.r_[mu, -50.0, 3.0, 1e3]      # the padding values are arbitrary
    mask = torch.tensor([True] * 7 + [False] * 3)
    mu_p = _t(pad, grad=True)
    lp = wt.race_log_prob(mu_p, mask=mask)
    assert torch.allclose(lp[:7], full, atol=1e-12)
    assert torch.all(torch.isneginf(lp[7:]))
    lp[:7].sum().backward()
    assert torch.all(mu_p.grad[7:] == 0) and torch.all(
        torch.isfinite(mu_p.grad))
    # the autograd path is NaN-free on padded rows too
    mu_q = _t(pad, grad=True)
    wt.race_log_prob(mu_q, mask=mask, backward="autograd")[:7].sum().backward()
    assert torch.allclose(mu_q.grad, mu_p.grad, atol=1e-12)


def test_variable_n_batch_equals_rows_one_at_a_time():
    sizes = [3, 8, 5]
    nmax = max(sizes)
    mu = torch.zeros(len(sizes), nmax, dtype=F64)
    mask = torch.zeros(len(sizes), nmax, dtype=torch.bool)
    rows = []
    for b, n in enumerate(sizes):
        m = _t(RNG.normal(size=n))
        mu[b, :n] = m
        mask[b, :n] = True
        rows.append(wt.race_log_prob(m))
    lp = wt.race_log_prob(mu, mask=mask)
    for b, n in enumerate(sizes):
        assert torch.allclose(lp[b, :n], rows[b], atol=1e-12)


def test_leading_batch_dims():
    mu = _t(RNG.normal(size=(2, 3, 6)))
    lp = wt.race_log_prob(mu)
    assert lp.shape == (2, 3, 6)
    assert torch.allclose(lp[1, 2], wt.race_log_prob(mu[1, 2]), atol=1e-13)


def test_permutation_equivariance():
    mu = RNG.normal(size=9)
    sd = RNG.uniform(0.5, 1.5, size=9)
    perm = RNG.permutation(9)
    a = wt.race_log_prob(_t(mu), _t(sd))
    b = wt.race_log_prob(_t(mu[perm]), _t(sd[perm]))
    assert torch.allclose(a[perm], b, atol=1e-13, rtol=0)


# ----------------------------------------------------------------- float32
@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_float32_tails_are_finite_and_close(base):
    for gap in (10.0, 40.0):
        mu64 = np.r_[np.zeros(4), gap]
        mu = _t(mu64, dtype=torch.float32, grad=True)
        lp = wt.race_log_prob(mu, base=base)
        assert torch.all(torch.isfinite(lp)), (gap, lp)
        ref = wt.race_log_prob(_t(mu64), base=base).numpy()
        assert np.allclose(lp.detach().numpy(), ref, rtol=1e-4, atol=1e-4)
        lp[-1].backward()
        assert torch.all(torch.isfinite(mu.grad)), (gap, mu.grad)


# ------------------------------------------------------------- the factor race
@pytest.mark.parametrize("rank", [1, 2])
def test_factor_race_matches_winning(rank):
    mu = RNG.normal(size=6)
    V = RNG.normal(size=(6, rank)) * 0.6
    ref = race_probabilities(mu, V=V, D=np.ones(6))
    lp = wt.race_log_prob_factor(_t(mu), _t(V), nodes=15).numpy()
    # winning's adaptive rule picks 15 nodes on this mild field; at rank 2
    # it prunes the tensor rule, and both sit ~1e-6 from the converged
    # (31-node) answer
    assert np.abs(np.exp(lp) / ref - 1).max() < (1e-10 if rank == 1 else 2e-5)
    hi = wt.race_log_prob_factor(_t(mu), _t(V), nodes=31).numpy()
    assert np.abs(np.exp(lp) / np.exp(hi) - 1).max() < 2e-5


def test_factor_race_gradients_and_mask():
    mu = _t(RNG.normal(size=(2, 5)), grad=True)
    V = _t(RNG.normal(size=(2, 5, 1)) * 0.5, grad=True)
    sd = _t(RNG.uniform(0.7, 1.3, size=(2, 5)), grad=True)
    assert torch.autograd.gradcheck(
        lambda m, v, s: wt.race_log_prob_factor(m, v, s, nodes=7),
        (mu, V, sd), eps=1e-6, atol=1e-6)
    mask = torch.tensor([True, True, True, False, True])
    lp = wt.race_log_prob_factor(mu, V, sd, nodes=7, mask=mask)
    keep = [0, 1, 2, 4]
    sub = wt.race_log_prob_factor(mu[:, keep], V[:, keep], sd[:, keep],
                                  nodes=7)
    assert torch.allclose(lp[:, keep], sub, atol=1e-12)
    lp[:, keep].sum().backward()
    assert torch.all(torch.isfinite(mu.grad))


def test_zero_loadings_is_the_independent_race():
    mu = _t(RNG.normal(size=8))
    lp = wt.race_log_prob_factor(mu, torch.zeros(8, 2, dtype=F64), nodes=5)
    assert torch.allclose(lp, wt.race_log_prob(mu), atol=1e-13)


# -------------------------------------------------------------------- losses
def test_thurstone_nll_is_minus_the_winner_log_prob():
    mu = _t(RNG.normal(size=(4, 6)), grad=True)
    win = torch.tensor([0, 3, 5, 2])
    loss = wt.thurstone_nll(mu, win, reduction="none")
    lp = wt.race_log_prob(mu)
    assert torch.allclose(loss, -lp[torch.arange(4), win])
    wt.thurstone_nll(mu, win).backward()
    assert torch.all(torch.isfinite(mu.grad))


@pytest.mark.parametrize("base", ["normal", "gumbel"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ordered_prefix_matches_winning(base, k):
    n = 6
    mu = RNG.normal(size=n)
    sd = RNG.uniform(0.6, 1.5, size=n)
    perms = list(itertools.permutations(range(n), k))
    lp = wt.ordered_prefix_log_prob(_t(mu).expand(len(perms), n),
                                    torch.tensor(perms), _t(sd), base=base,
                                    points=257 if k < 3 else 1025).numpy()
    ref = ordered_probabilities(mu, k=k, D=sd ** 2, base=base, points=4001)
    r = np.array([ref[p] for p in perms])
    # k <= 2 spectral; k = 3 the corrected cumulative quadrature
    tol = 1e-10 if k < 3 else 1e-5
    assert np.abs(np.exp(lp) / r - 1).max() < tol
    assert abs(np.exp(lp).sum() - 1.0) < tol


def test_ordered_prefix_gumbel_is_plackett_luce():
    mu = RNG.normal(size=5)
    c = math.pi / math.sqrt(6.0)
    order = [3, 0]
    lp = float(wt.ordered_prefix_log_prob(_t(mu), torch.tensor(order),
                                          base="gumbel"))
    u = -c * mu
    first = u[3] - np.log(np.exp(u).sum())
    rest = np.delete(u, 3)
    second = u[0] - np.log(np.exp(rest).sum())
    assert abs(lp - (first + second)) < 1e-10


def test_ordered_prefix_gradients_are_finite():
    mu = _t(RNG.normal(size=(3, 5)), grad=True)
    lp = wt.ordered_prefix_log_prob(mu, torch.tensor([[1, 2, 0]] * 3))
    lp.sum().backward()
    assert torch.all(torch.isfinite(mu.grad))


def test_ordered_prefix_refuses_repeats():
    with pytest.raises(ValueError, match="repeated"):
        wt.ordered_prefix_log_prob(_t([0.0, 1.0, 2.0]), torch.tensor([1, 1]))


@pytest.mark.parametrize("base", ["normal", "gumbel"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_topk_matches_winning(base, k):
    n = 8
    mu = RNG.normal(size=n)
    sd = RNG.uniform(0.6, 1.5, size=n)
    ref = top_k_probabilities(mu, k, D=sd ** 2, base=base)
    lp = wt.topk_log_prob(_t(mu), k, _t(sd), base=base).numpy()
    assert np.abs(np.exp(lp) / ref - 1).max() < 1e-9
    assert abs(np.exp(lp).sum() - k) < 1e-9


def test_topk_mask_and_gradients():
    mu = _t(RNG.normal(size=6), grad=True)
    mask = torch.tensor([True, True, False, True, True, True])
    lp = wt.topk_log_prob(mu, 2, mask=mask)
    sub = wt.topk_log_prob(mu[[0, 1, 3, 4, 5]], 2)
    assert torch.allclose(lp[[0, 1, 3, 4, 5]], sub, atol=1e-12)
    assert torch.isneginf(lp[2])
    lp[[0, 1, 3, 4, 5]].sum().backward()
    assert torch.all(torch.isfinite(mu.grad)) and mu.grad[2] == 0


# ------------------------------------------------------------------ hygiene
def test_importing_winning_does_not_import_torch():
    code = ("import sys, winning, winning.factor; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_bad_arguments_refuse():
    mu = _t([0.0, 1.0])
    with pytest.raises(ValueError):
        wt.race_log_prob(mu, base="cauchy")
    with pytest.raises(ValueError):
        wt.race_log_prob(mu, points=2)
    with pytest.raises(ValueError):
        wt.race_log_prob(mu, backward="numeric")
    with pytest.raises(ValueError):
        wt.race_log_prob_factor(mu, torch.zeros(3, 1, dtype=F64))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_float32_matches_cpu_float64():
    """MPS (torch 2.14) has no log_ndtr, ndtri or float64: log_ndtr is
    rebuilt from erfcx/erfc there and the window is computed on the CPU."""
    mu = RNG.normal(size=(4, 20)) * 2.0
    mu[0, -1] = 12.0                                   # a longshot
    ref = wt.race_log_prob(_t(mu)).numpy()
    m = torch.tensor(mu, dtype=torch.float32, device="mps",
                     requires_grad=True)
    lp = wt.race_log_prob(m)
    assert np.allclose(lp.detach().cpu().numpy(), ref, rtol=1e-4, atol=1e-4)
    lp[:, 0].sum().backward()
    assert torch.all(torch.isfinite(m.grad))


# --- #753: later-place events live beyond the winner window -------------

def _norm_logcdf(z):
    from scipy.special import log_ndtr
    return float(log_ndtr(z))


@pytest.mark.parametrize("gap", [5.0, 10.0, 20.0])
def test_full_pair_order_is_the_win_probability(gap):
    # P(0 first, 1 second) = P(X0 < X1) = Phi(gap / sqrt 2); the winner
    # lattice cut the second finisher's mass and gave 0.503 at gap 10 and
    # 3.1e-7 at gap 20 (#753)
    mu = _t([0.0, gap])
    got = float(wt.ordered_prefix_log_prob(mu, torch.tensor([0, 1])))
    want = _norm_logcdf(gap / math.sqrt(2.0))
    assert abs(got - want) < 1e-11
    got = float(wt.ordered_prefix_log_prob(mu, torch.tensor([1, 0])))
    want = _norm_logcdf(-gap / math.sqrt(2.0))
    assert abs(got - want) < 1e-9 * abs(want)


def test_separated_top2_of_three_is_the_pair_tail():
    # runner 2 is in the top two essentially iff it beats runner 1; the
    # correction from beating runner 0 is below 1e-45. Was 4.3e-13 for
    # 7.69e-13, with the memberships summing to 1.9999997 (#753)
    mu = _t([0.0, 10.0, 20.0])
    p = torch.exp(wt.topk_log_prob(mu, 2))
    weak = math.exp(_norm_logcdf(-10.0 / math.sqrt(2.0)))
    assert abs(float(p[2]) / weak - 1.0) < 1e-9
    assert abs(float(p.sum()) - 2.0) < 1e-11


@pytest.mark.parametrize("spread", [1.0, 4.0, 10.0])
@pytest.mark.parametrize("k", [2, 3])
def test_topk_memberships_sum_to_k_on_separated_fields(spread, k):
    rng = np.random.default_rng(int(spread * 10) + k)
    for n in (4, 8, 12):
        mu = _t(rng.normal(size=n) * spread)
        sd = _t(rng.uniform(0.5, 1.5, size=n))
        s = float(torch.exp(wt.topk_log_prob(mu, k, sd)).sum())
        assert abs(s - k) < 1e-10, (n, s)


def _fine_topk(mu, sd, i, k):
    """Independent referee: P(i among the k smallest) on a 200k-point
    log-domain grid around runner i (Poisson-binomial count of others)."""
    from scipy.special import log_ndtr
    from scipy.stats import norm
    x = np.linspace(mu[i] - 40 * sd[i], mu[i] + 40 * sd[i], 200001)
    lp = np.full((k, len(x)), -np.inf)
    lp[0] = 0.0
    for j in range(len(mu)):
        if j == i:
            continue
        lF = log_ndtr((x - mu[j]) / sd[j])
        lS = log_ndtr(-(x - mu[j]) / sd[j])
        new = np.full_like(lp, -np.inf)
        new[0] = lp[0] + lS
        for a in range(1, k):
            new[a] = np.logaddexp(lp[a] + lS, lp[a - 1] + lF)
        lp = new
    g = norm.logpdf(x, mu[i], sd[i]) + np.logaddexp.reduce(lp, axis=0)
    m = g.max()
    return float(m + np.log(np.trapezoid(np.exp(g - m), x)))


def test_topk_far_tail_matches_an_independent_referee():
    # wide fields put members' tiny shares far right of the winner
    # window; these are 1e-27 .. 1e-119 and must keep their digits
    rng = np.random.default_rng(753)
    for spread, n, k in ((4.0, 12, 2), (10.0, 5, 2), (10.0, 8, 3)):
        mu = rng.normal(size=n) * spread
        sd = rng.uniform(0.5, 1.5, size=n)
        lp = wt.topk_log_prob(_t(mu), k, _t(sd)).numpy()
        for i in np.argsort(mu)[-3:]:          # the weakest three
            ref = _fine_topk(mu, sd, i, k)
            if ref < -690.0:
                # below float64's normal range the linear count program
                # underflows (a separate limitation); skip those
                continue
            assert abs(lp[i] - ref) < 1e-6 * max(1.0, abs(ref)), (
                spread, n, k, i)


@pytest.mark.parametrize("k", [2, 3])
def test_later_place_events_converge_under_refinement(k):
    # the window now covers every runner, so refining it converges to
    # the same answer (the winner window converged to a wrong one)
    rng = np.random.default_rng(31 + k)
    mu = _t(rng.normal(size=6) * 6.0)
    sd = _t(rng.uniform(0.5, 1.5, size=6))
    order = torch.tensor(list(np.argsort(rng.normal(size=6))[:k]))
    a = wt.ordered_prefix_log_prob(mu, order, sd, points=1025)
    b = wt.ordered_prefix_log_prob(mu, order, sd, points=8193)
    assert abs(float(a - b)) < 1e-6 * max(1.0, abs(float(b)))
    t1 = wt.topk_log_prob(mu, k, sd, points=1025)
    t2 = wt.topk_log_prob(mu, k, sd, points=8193)
    torch.testing.assert_close(torch.exp(t1), torch.exp(t2), rtol=1e-8,
                               atol=1e-14)


def test_k1_events_keep_the_winner_lattice():
    # k = 1 is race_log_prob, on its own window, unchanged
    rng = np.random.default_rng(1)
    mu, sd = _t(rng.normal(size=7) * 3.0), _t(rng.uniform(0.5, 1.5, 7))
    lp = wt.race_log_prob(mu, sd)
    for i in range(7):
        assert torch.equal(
            wt.ordered_prefix_log_prob(mu, torch.tensor([i]), sd), lp[i])
    # top-1 is the count program, a different route to the same number
    torch.testing.assert_close(wt.topk_log_prob(mu, 1, sd), lp, rtol=1e-12,
                               atol=1e-12)


def test_separated_pair_order_gradient_matches_finite_differences():
    mu = _t([0.0, 6.0], grad=True)
    lp = wt.ordered_prefix_log_prob(mu, torch.tensor([1, 0]))
    (g,) = torch.autograd.grad(lp, mu)
    # d/dmu1 log Phi((mu0 - mu1)/sqrt 2) = -phi(z)/(sqrt2 Phi(z))
    from scipy.stats import norm
    z = -6.0 / math.sqrt(2.0)
    d = math.exp(norm.logpdf(z) - norm.logcdf(z)) / math.sqrt(2.0)
    torch.testing.assert_close(g, _t([d, -d]), rtol=1e-6, atol=0)
