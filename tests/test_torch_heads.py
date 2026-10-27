"""winning.torch: Thurstone heads for PyTorch.

Skipped where torch is absent (CI has no torch job). Parity is against
winning.factor at its defaults; the two lattices differ (winning bisects
the winner distribution's quantiles, winning.torch takes a closed-form
per-row window, docs/design/torch.md), so agreement is to the forward's
own accuracy, ~1e-12 relative, not bitwise.
"""
import itertools
import warnings
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

    def rel_err(ref, lp=lp):
        ok = ref > 1e-290
        return np.abs(np.exp(lp[ok]) / ref[ok] - 1.0).max()

    # relative, even for 1e-176 longshots: both lattices resolve them
    err = rel_err(race_probabilities(mu, D=sd ** 2, base=base))
    if err >= 1e-10 and base != "normal":
        # winning's bulk window post-checks rare winners for the normal
        # base only: on gumbel rare_middle (hetero) its default is 8e-6
        # off. The converged span lattice is the reference instead; it
        # agrees with mpmath quadrature to 5e-13 there (its floor). Raw
        # against it, this module is 3.2e-10 off at 257 points: its OWN
        # spacing error on the gumbel's sharp flank (dx = 0.3 min sd),
        # since 513 points give 3e-14. So: the default within that
        # measured 3.2e-10, and the refined lattice to the span floor.
        span = race_probabilities(mu, D=sd ** 2, base=base, window="span",
                                  points=20001)
        assert rel_err(span) < 1e-9, (name, rel_err(span))
        lp513 = wt.race_log_prob(_t(mu), _t(sd), base=base,
                                 points=513).numpy()
        err = rel_err(span, lp513)
        assert err < 1e-12, (name, err)
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


# ------------------------------------- #753: later places on separated fields
# ordered_prefix_log_prob / topk_log_prob used the WINNER lattice, whose
# right edge stops where every winner integrand is negligible. A later
# place has its mass near its own runner, often far to the right of
# that: P(0 then 1) at mu = [0, 10] came out 0.503 (exact 1 - 7.7e-13).
# References below are exact: closed forms, or a one-dimensional
# log-domain quadrature on 200001 points (spectral for these smooth
# integrands, ~1e-14), independent of winning's lattices.
from scipy.special import log_ndtr as _log_ndtr  # noqa: E402
from scipy.special import logsumexp as _lse  # noqa: E402

GAPS = [5.0, 10.0, 20.0, 40.0]
SDS2 = [(1.0, 1.0), (0.3, 1.0), (1.0, 0.3), (2.0, 0.5)]
SDS3 = [(1.0, 1.0, 1.0), (0.3, 1.0, 2.0), (2.0, 1.0, 0.3)]


def _ref_terms(mu, sd):
    x = np.linspace((mu - 40 * sd).min(), (mu + 40 * sd).max(), 200001)
    z = (x[None] - mu[:, None]) / sd[:, None]
    lf = -0.5 * z * z - 0.5 * math.log(2 * math.pi) - np.log(sd)[:, None]
    return lf, _log_ndtr(z), _log_ndtr(-z), math.log(x[1] - x[0])


def _ref_order3(mu, sd, o):
    """log P(X_a < X_b < X_c) = log int f_b F_a S_c."""
    a, b, c = o
    lf, lF, lS, ldx = _ref_terms(mu, sd)
    return _lse(lf[b] + lF[a] + lS[c]) + ldx


def _ref_top2_of3(mu, sd, i):
    """log P(i among the best two of three) = log int f_i (1 - F_j F_l)."""
    j, m = [r for r in range(3) if r != i]
    lf, lF, _, ldx = _ref_terms(mu, sd)
    with np.errstate(divide="ignore"):
        w = np.log(-np.expm1(lF[j] + lF[m]))
    return _lse(lf[i] + w) + ldx


def _rel(got, ref):
    return abs(math.expm1(float(got) - float(ref)))


@pytest.mark.parametrize("gap", GAPS)
@pytest.mark.parametrize("sds", SDS2)
def test_753_two_runner_order_is_the_pair_probability(gap, sds):
    """Full order of a pair = P(X_a < X_b), relative, both orders (the
    rare order is 1e-5 .. 1e-176 here)."""
    mu = np.array([0.0, gap])
    sd = np.array(sds)
    for o in [(0, 1), (1, 0)]:
        got = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o), _t(sd))
        ref = _log_ndtr((mu[o[1]] - mu[o[0]]) / math.hypot(*sd))
        assert _rel(got, ref) < 1e-10, (o, float(got), ref)


@pytest.mark.parametrize("gap", GAPS)
@pytest.mark.parametrize("sds", SDS3)
def test_753_three_runner_prefixes_on_separated_fields(gap, sds):
    """k = 2 prefixes of three runners (each is a full order, spectral on
    the lattice) to 1e-10 relative, down to 1e-300; k = 3 to 1e-4 for
    events above 1e-12 (the cumulative quadrature; see the next test)."""
    mu = np.array([0.0, gap, 2 * gap])
    sd = np.array(sds)
    for o in itertools.permutations(range(3)):
        ref = _ref_order3(mu, sd, o)
        if ref < -690:
            continue
        k2 = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o[:2]), _t(sd))
        assert _rel(k2, ref) < 1e-10, (o, float(k2), ref)
        if ref > math.log(1e-12):
            k3 = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o), _t(sd))
            assert _rel(k3, ref) < 1e-4, (o, float(k3), ref)


def test_753_k3_prefix_converges_under_refinement():
    """k = 3 is a cumulative trapezoid with an Euler-Maclaurin end
    correction across the whole (now wider) lattice: fourth order, so 4x
    the points cuts the error ~256x, here on rare events (1e-12 .. 1e-47)
    of a separated field, where the error is largest."""
    mu = np.array([0.0, 10.0, 20.0])
    sd = np.ones(3)
    for o in [(0, 2, 1), (1, 2, 0), (2, 1, 0)]:
        ref = _ref_order3(mu, sd, o)
        e = [_rel(wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o), _t(sd),
                                             points=p), ref)
             for p in (257, 1025)]
        assert e[1] < 1e-5 and e[1] < e[0] / 50, (o, e)


def test_753_far_ahead_outsider_keeps_the_prefix_relative():
    """A non-prefix runner 40 sd ahead must finish behind the prefix: the
    event (1e-235) lives far LEFT, which the win lattice's relative left
    edge already covers."""
    mu = np.array([0.0, 0.0, -40.0])
    sd = np.ones(3)
    for o in itertools.permutations(range(3)):
        ref = _ref_order3(mu, sd, o)
        got = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o[:2]), _t(sd))
        assert _rel(got, ref) < 1e-10, (o, float(got), ref)


@pytest.mark.parametrize("gap", GAPS)
@pytest.mark.parametrize("sds", SDS3)
def test_753_top2_of_three_on_separated_fields(gap, sds):
    """The issue's fixture ([0, 10, 20]: weak runner 4.3e-13 for 7.7e-13)
    and its relatives, relative to exact, and sum = 2."""
    mu = np.array([0.0, gap, 2 * gap])
    sd = np.array(sds)
    lp = wt.topk_log_prob(_t(mu), 2, _t(sd)).numpy()
    for i in range(3):
        ref = _ref_top2_of3(mu, sd, i)
        if ref > -690:
            assert _rel(lp[i], ref) < 1e-10, (i, lp[i], ref)
    assert abs(np.exp(lp).sum() - 2.0) < 1e-11


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("gap", [1.0] + GAPS)
def test_753_full_orderings_sum_to_one(n, gap):
    mu = gap * np.arange(n, dtype=float)
    sd = np.linspace(0.5, 1.5, n)
    perms = list(itertools.permutations(range(n)))
    lp = wt.ordered_prefix_log_prob(_t(mu).expand(len(perms), n),
                                    torch.tensor(perms), _t(sd))
    assert abs(torch.exp(lp).sum().item() - 1.0) < 1e-7
    # and every k = 2 prefix family sums to one (spectral)
    perms2 = list(itertools.permutations(range(n), 2))
    lp2 = wt.ordered_prefix_log_prob(_t(mu).expand(len(perms2), n),
                                     torch.tensor(perms2), _t(sd))
    assert abs(torch.exp(lp2).sum().item() - 1.0) < 1e-11


@pytest.mark.parametrize("base", ["normal", "gumbel"])
@pytest.mark.parametrize("gap", GAPS)
def test_753_topk_memberships_sum_to_k(base, gap):
    mu = gap * np.array([0.0, 1.0, 2.0, 3.0, 0.5])
    sd = np.array([1.0, 0.4, 2.0, 1.0, 0.7])
    # the min-Gumbel density falls double-exponentially on its right, so
    # at the default spacing it resolves to ~1e-10, not ~1e-13
    tol = 1e-10 if base == "normal" else 1e-9
    for k in range(1, 5):
        lp = wt.topk_log_prob(_t(mu), k, _t(sd), base=base)
        assert abs(torch.exp(lp).sum().item() - k) < tol, k


def _random_fields(m, seed):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(m):
        n = int(rng.integers(3, 7))
        out.append(([1, 3, 10, 25][t % 4] * rng.normal(size=n),
                    rng.uniform(0.3, 2.0, size=n)))
    return out


@pytest.mark.parametrize("field", range(16))
def test_753_random_fields_agree_with_winning_factor(field):
    """Ordinary to well-separated random fields (spread 1..25 sd, sd 0.3
    .. 2): k = 2 prefixes and top-k for k = 1..3 agree with
    winning.factor.ordered_probabilities / top_k_probabilities (which
    price these fields correctly; they size their lattice to every
    runner's bulk). On main the k = 2 prefixes were off by up to 100%
    (2e-4 at spread 1)."""
    mu, sd = _random_fields(16, 753)[field]
    n = len(mu)
    perms = list(itertools.permutations(range(n), 2))
    lp = wt.ordered_prefix_log_prob(_t(mu).expand(len(perms), n),
                                    torch.tensor(perms), _t(sd)).numpy()
    ref = ordered_probabilities(mu, k=2, D=sd ** 2, points=4001)
    r = np.array([ref[p] for p in perms])
    ok = r > 1e-280
    assert np.abs(np.expm1(lp[ok] - np.log(r[ok]))).max() < 1e-10
    for k in range(1, min(3, n - 1) + 1):
        lt = wt.topk_log_prob(_t(mu), k, _t(sd)).numpy()
        rt = top_k_probabilities(mu, k, D=sd ** 2)
        ok = rt > 1e-280
        assert np.abs(np.expm1(lt[ok] - np.log(rt[ok]))).max() < 1e-10, k


def test_753_ordinary_hetero_k3_prefix_is_right():
    """Not only separated fields: with sd from 0.3 to 2 the winner lattice
    clipped the slow runner's bulk, and k = 3 full orders of ORDINARY
    three-runner fields were off by up to 88% on main."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(12):
        mu = rng.normal(size=3)
        sd = rng.uniform(0.3, 2.0, 3)
        for o in itertools.permutations(range(3)):
            ref = _ref_order3(mu, sd, o)
            got = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o), _t(sd))
            worst = max(worst, _rel(got, ref))
    assert worst < 1e-4          # k = 3 quadrature; main: 0.88


@pytest.mark.parametrize("gap", GAPS)
def test_753_gumbel_prefixes_are_plackett_luce_when_separated(gap):
    """Under the min-Gumbel base the prefix law is Plackett-Luce exactly.
    The wider lattice reaches x where a prefix runner's log S = -e^u is
    ~-1e17; rest = sum(all) - sum(prefix) then lost every digit, so the
    non-prefix runners are now summed directly. (1e-8: the rarest of
    these events, 1e-67 at gap 40, resolves to 1.3e-9 at 257 points and
    1e-13 at 1025; the Gumbel's right flank is sharper than its sd.)"""
    c = math.pi / math.sqrt(6.0)
    mu = np.array([0.0, gap, -gap, 0.5])

    def pl(o):
        u = -c * mu
        rem, s = list(range(len(mu))), 0.0
        for i in o:
            s += u[i] - _lse(u[rem])
            rem.remove(i)
        return s

    for o in itertools.permutations(range(4), 2):
        got = wt.ordered_prefix_log_prob(_t(mu), torch.tensor(o),
                                         base="gumbel")
        assert _rel(got, pl(o)) < 1e-8, (o, float(got), pl(o))


def test_753_batch_of_mixed_fields_matches_rows():
    """One lattice length serves a batch (the largest any row needs), so a
    separated row in a batch of ordinary rows is priced right, and every
    row equals its own call to quadrature accuracy."""
    fields = [(np.array([0.0, 0.5, 1.0]), np.ones(3)),
              (np.array([0.0, 10.0, 20.0]), np.array([0.3, 1.0, 2.0])),
              (np.array([0.0, 40.0, 1.0]), np.ones(3))]
    mu = _t(np.stack([f[0] for f in fields]))
    sd = _t(np.stack([f[1] for f in fields]))
    order = torch.tensor([[1, 0], [0, 2], [2, 1]])
    lp = wt.ordered_prefix_log_prob(mu, order, sd)
    lt = wt.topk_log_prob(mu, 2, sd)
    for r in range(3):
        one = wt.ordered_prefix_log_prob(mu[r], order[r], sd[r])
        assert _rel(lp[r], one) < 1e-11
        assert np.abs(np.expm1((lt[r] - wt.topk_log_prob(
            mu[r], 2, sd[r])).numpy())).max() < 1e-11
        o3 = tuple(order[r].tolist()) + tuple(
            {0, 1, 2} - set(order[r].tolist()))
        assert _rel(lp[r], _ref_order3(fields[r][0], fields[r][1], o3)) \
            < 1e-10


def test_753_padding_does_not_move_the_event_lattice():
    """Masked runners are out of the window and the spacing rule: a
    padded entry with an extreme mu / tiny sd changes nothing."""
    mu = _t([0.0, 10.0, 500.0, 20.0])
    sd = _t([1.0, 1.0, 1e-3, 0.5])
    mask = torch.tensor([True, True, False, True])
    sub_mu, sub_sd = mu[[0, 1, 3]], sd[[0, 1, 3]]
    lp = wt.ordered_prefix_log_prob(mu, torch.tensor([3, 1]), sd, mask=mask)
    ref = wt.ordered_prefix_log_prob(sub_mu, torch.tensor([2, 1]), sub_sd)
    assert torch.equal(lp, ref)
    lt = wt.topk_log_prob(mu, 2, sd, mask=mask)
    assert torch.equal(lt[[0, 1, 3]], wt.topk_log_prob(sub_mu, 2, sub_sd))
    assert torch.isneginf(lt[2])


def test_753_lattice_size_rule_and_cap():
    from winning.torch.losses import (_EVENT_CAP, _EVENT_SPACING,
                                      _event_lattice_rows)
    from winning.torch.race import _as_base, _lattice_rows
    fam = _as_base("normal")

    def size(mu, sd, points=257):
        return _event_lattice_rows(_t(mu)[None], _t(sd)[None], None, fam,
                                   points, 1e-12).shape[-1]

    # ordinary: points= unchanged
    assert size([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == 257
    # separated, narrow runner: spacing <= _EVENT_SPACING * min sd
    x = _event_lattice_rows(_t([0.0, 40.0, 80.0])[None],
                            _t([0.3, 1.0, 2.0])[None], None, fam, 257, 1e-12)
    assert x.shape[-1] > 257
    assert float(x[0, 1] - x[0, 0]) <= _EVENT_SPACING * 0.3 + 1e-12
    # covers every runner's bulk (the win lattice stops near 40 + 7)
    assert float(x[0, -1]) >= 80.0 + 7.0 * 2.0
    win = _lattice_rows(_t([0.0, 40.0, 80.0])[None],
                        _t([0.3, 1.0, 2.0])[None], None, fam, 257, 1e-12)
    assert float(win[0, -1]) < 60.0
    # capped (and, past ~1 sd spacing, saying so), unless points= asks
    # for more
    with pytest.warns(RuntimeWarning, match="cannot resolve the narrowest"):
        assert size([0.0, 5000.0], [1.0, 1.0]) == _EVENT_CAP
    assert size([0.0, 5000.0], [1.0, 1.0], points=6001) == 6001


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_753_gradcheck_on_a_separated_field(base):
    mu = _t([0.0, 10.0, 20.0], grad=True)
    sd = _t([1.0, 0.5, 2.0], grad=True)
    assert torch.autograd.gradcheck(
        lambda m, s: wt.ordered_prefix_log_prob(m, torch.tensor([1, 0]), s,
                                                base=base),
        (mu, sd), eps=1e-6, atol=1e-6)
    assert torch.autograd.gradcheck(
        lambda m, s: wt.topk_log_prob(m, 2, s, base=base),
        (mu, sd), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_753_float32_separated_fields(base):
    """float32: the likely events are right (main: 0.503 for ~1), sums
    hold, gradients are finite."""
    mu = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float32,
                      requires_grad=True)
    p = torch.exp(wt.ordered_prefix_log_prob(mu[:2], torch.tensor([0, 1]),
                                             base=base))
    c = math.pi / math.sqrt(6.0)
    exact = (math.exp(_log_ndtr(10.0 / math.sqrt(2.0))) if base == "normal"
             else 1.0 / (1.0 + math.exp(-10.0 * c)))
    assert abs(p.item() - exact) < 1e-6
    lt = wt.topk_log_prob(mu, 2, base=base)
    assert abs(torch.exp(lt).sum().item() - 2.0) < 1e-5
    lp64 = wt.topk_log_prob(mu.detach().double(), 2, base=base)
    assert torch.allclose(torch.exp(lt).double(), torch.exp(lp64), atol=1e-6)
    (lt.sum() + wt.ordered_prefix_log_prob(
        mu, torch.tensor([2, 1, 0]), base=base)).backward()
    assert torch.all(torch.isfinite(mu.grad))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_753_mps_event_lattice_matches_cpu():
    """The event lattice is sized on the CPU (one sync) for MPS too."""
    mu = np.array([[0.0, 10.0, 20.0], [0.0, 0.5, 1.0]])
    m = torch.tensor(mu, dtype=torch.float32, device="mps",
                     requires_grad=True)
    order = torch.tensor([[1, 0], [2, 1]], device="mps")
    a = wt.ordered_prefix_log_prob(m, order)
    b = wt.topk_log_prob(m, 2)
    ra = wt.ordered_prefix_log_prob(_t(mu), order.cpu())
    rb = wt.topk_log_prob(_t(mu), 2)
    assert np.allclose(a.detach().cpu().numpy(), ra.numpy(), rtol=1e-4,
                       atol=1e-4)
    assert np.allclose(torch.exp(b).detach().cpu().numpy(),
                       torch.exp(rb).numpy(), atol=1e-5)
    (a.sum() + b.sum()).backward()
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


def _fine_topk(mu, sd, i, k, width=40.0):
    """Independent referee: P(i among the k smallest) on a 200k-point
    log-domain grid of +-width sd around runner i (Poisson-binomial count
    of others)."""
    from scipy.special import log_ndtr
    from scipy.stats import norm
    x = np.linspace(mu[i] - width * sd[i], mu[i] + width * sd[i], 200001)
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
            # below float64's normal range too, since the count program
            # went log-domain (#863)
            ref = _fine_topk(mu, sd, i, k)
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
        torch.testing.assert_close(
            wt.ordered_prefix_log_prob(mu, torch.tensor([i]), sd), lp[i],
            rtol=1e-12, atol=1e-12)
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


# --- #756: heterogeneous scales ------------------------------------------

@pytest.mark.parametrize("ratio", [10.0, 30.0, 100.0, 200.0])
def test_equal_means_any_scales_give_exact_identities(ratio):
    # two equal-mean normals: each full order is 1/2, each is in the top
    # two with probability 1. The fixed-count wide lattice gave 1.334 for
    # a top-2 membership at ratio 100 (#756)
    mu = _t([0.0, 0.0])
    sd = _t([1.0, ratio])
    p01 = torch.exp(wt.ordered_prefix_log_prob(mu, torch.tensor([0, 1]), sd))
    p10 = torch.exp(wt.ordered_prefix_log_prob(mu, torch.tensor([1, 0]), sd))
    assert abs(float(p01) - 0.5) < 1e-9 and abs(float(p10) - 0.5) < 1e-9
    top2 = torch.exp(wt.topk_log_prob(mu, 2, sd))
    torch.testing.assert_close(top2, _t([1.0, 1.0]), rtol=0, atol=1e-9)


def test_unresolvable_scale_ratio_warns_and_points_resolves_it():
    # past the lattice cap the call warns instead of returning a quiet
    # probability above one; points= large enough resolves it
    mu, sd = _t([0.0, 0.0]), _t([1.0, 1000.0])
    with pytest.warns(RuntimeWarning, match="cannot resolve the narrowest"):
        wt.topk_log_prob(mu, 2, sd)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        top2 = torch.exp(wt.topk_log_prob(mu, 2, sd, points=60001))
    torch.testing.assert_close(top2, _t([1.0, 1.0]), rtol=0, atol=1e-8)


def test_event_lattice_is_quiet_on_ordinary_heads():
    # the warning must not fire on everyday inputs: sd within 0.3..3
    import warnings as _w
    rng = np.random.default_rng(756)
    with _w.catch_warnings():
        _w.simplefilter("error")
        for _ in range(40):
            n = int(rng.integers(2, 20))
            mu = _t(rng.normal(size=n) * rng.uniform(0.5, 8.0))
            sd = _t(rng.uniform(0.3, 3.0, size=n))
            wt.topk_log_prob(mu, min(2, n - 1) or 1, sd)
            wt.ordered_prefix_log_prob(mu, torch.tensor([0, n - 1]), sd)


# --- #760: masked padding is algebraically absent -------------------------

_BAD_PADDING = [(1.0, 0.0), (1.0, -2.0), (float("nan"), 1.0),
                (1.0, float("nan")), (float("inf"), 1.0), (1.0, float("inf"))]


@pytest.mark.parametrize("pad_mu,pad_sd", _BAD_PADDING)
@pytest.mark.parametrize("path", ["analytic", "autograd", "prefix2",
                                  "top2", "factor"])
def test_invalid_padding_gives_the_unpadded_race_and_zero_gradient(
        path, pad_mu, pad_sd):
    # sd = 0 or NaN in a masked slot gave NaN gradients (#760)
    mu = _t([0.0, 0.7, pad_mu, 0.3], grad=True)
    sd = _t([1.0, 0.8, pad_sd, 1.3], grad=True)
    mask = torch.tensor([True, True, False, True])
    mu0 = _t([0.0, 0.7, 0.3])
    sd0 = _t([1.0, 0.8, 1.3])
    V = _t([[0.4], [-0.2], [float("nan")], [0.1]], grad=True)

    def run(m, s, msk, v=None):
        if path in ("analytic", "autograd"):
            return wt.race_log_prob(m, s, mask=msk, backward=path)
        if path == "prefix2":
            o = torch.tensor([0, 3]) if msk is not None else torch.tensor(
                [0, 2])
            return wt.ordered_prefix_log_prob(m, o, s, mask=msk)
        if path == "top2":
            return wt.topk_log_prob(m, 2, s, mask=msk)
        return wt.race_log_prob_factor(m, v, s, mask=msk)

    out = run(mu, sd, mask, V)
    ref = run(mu0, sd0, None, _t([[0.4], [-0.2], [0.1]]))
    got = out if out.dim() == 0 else out[mask]
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    loss = got.sum()
    loss.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sd.grad).all()
    assert float(mu.grad[2]) == 0.0 and float(sd.grad[2]) == 0.0
    if path == "factor":
        assert torch.isfinite(V.grad).all() and float(V.grad[2, 0]) == 0.0


def test_order_selecting_padding_or_out_of_range_refuses():
    # a masked runner gathered as a finisher priced a "probability" of 7.1
    # and sent gradient to padding (#760, second residual)
    mu, sd = _t([0.0, 1.0, 2.0]), _t([1.0, 1.0, 1.0])
    mask = torch.tensor([True, False, True])
    for order in ([1], [0, 1], [1, 0]):
        with pytest.raises(ValueError, match="masked"):
            wt.ordered_prefix_log_prob(mu, torch.tensor(order), sd,
                                       mask=mask)
    with pytest.raises(ValueError, match="outside"):
        wt.ordered_prefix_log_prob(mu, torch.tensor([0, 3]), sd)
    # batched: a mask that pads a different runner per row
    mu2 = _t([[0.0, 1.0, 2.0], [0.5, 0.0, 1.0]])
    m2 = torch.tensor([[True, True, False], [True, True, True]])
    with pytest.raises(ValueError, match="masked"):
        wt.ordered_prefix_log_prob(mu2, torch.tensor([0, 2]), mask=m2)
    out = wt.ordered_prefix_log_prob(mu2, torch.tensor([0, 1]), mask=m2)
    assert torch.isfinite(out).all() and bool((out < 0).all())


@pytest.mark.parametrize("bad,match", [
    (dict(sd=[1.0, 0.0]), "sd must be finite and positive"),
    (dict(sd=[1.0, -1.0]), "sd must be finite and positive"),
    (dict(mu=[0.0, float("nan")]), "non-finite"),
    (dict(mask=[False, False]), "at least one valid runner"),
])
def test_invalid_active_inputs_refuse(bad, match):
    mu = _t(bad.get("mu", [0.0, 1.0]))
    sd = _t(bad.get("sd", [1.0, 1.0]))
    mask = torch.tensor(bad["mask"]) if "mask" in bad else None
    with pytest.raises(ValueError, match=match):
        wt.race_log_prob(mu, sd, mask=mask)


# ------------------------------------------- #863: log-domain count program
def _fd_fine_topk(mu, sd, i, k, h=1e-4):
    g = np.zeros(len(mu))
    for j in range(len(mu)):
        up, dn = mu.copy(), mu.copy()
        up[j] += h
        dn[j] -= h
        g[j] = (_fine_topk(up, sd, i, k) - _fine_topk(dn, sd, i, k)) / (2 * h)
    return g


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_863_rare_top1_is_the_race_value_and_gradient(dtype):
    # the linear count program clamped at finfo.tiny: float32 top-1 of a
    # 20-sd longshot came out -87.3 (log tiny) for -170.1, with ~zero
    # gradient; float64 did the same at a gap of 60 (-708 for -1454)
    gaps = (20.0, 60.0) if dtype == F64 else (20.0,)
    tol = 1e-12 if dtype == F64 else 1e-6
    for gap in gaps:
        mu = _t([0.0, 0.0, 0.0, 0.0, gap], dtype=dtype, grad=True)
        top = wt.topk_log_prob(mu, 1)[-1]
        (g_top,) = torch.autograd.grad(top, mu)
        m64 = mu.detach().to(F64).requires_grad_()
        race = wt.race_log_prob(m64)[-1]
        (g_race,) = torch.autograd.grad(race, m64)
        top, race = float(top.detach()), float(race.detach())
        assert abs(top / race - 1.0) < tol, (gap, top)
        torch.testing.assert_close(g_top.to(F64), g_race, rtol=10 * tol,
                                   atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, F64])
@pytest.mark.parametrize("k", [2, 3])
def test_863_rare_topk_matches_a_log_domain_referee(dtype, k):
    mu_np = np.array([0.0, 0.0, 0.0, 0.0, 20.0])
    sd_np = np.ones(5)
    ref = _fine_topk(mu_np, sd_np, 4, k)              # -156.9 / -137.7
    g_ref = _fd_fine_topk(mu_np, sd_np, 4, k)
    mu = _t(mu_np, dtype=dtype, grad=True)
    lp = wt.topk_log_prob(mu, k)[-1]
    (g,) = torch.autograd.grad(lp, mu)
    tol = 1e-9 if dtype == F64 else 1e-6
    lp = float(lp.detach())
    assert abs(lp / ref - 1.0) < tol, (lp, ref)
    # the gradient is O(10) per runner; finite differences of the
    # referee are good to ~1e-6
    np.testing.assert_allclose(g.to(F64).numpy(), g_ref, rtol=1e-5,
                               atol=1e-5)


def test_863_below_float64_normal_range_keeps_its_digits():
    # memberships far below 1e-308 used to floor at log(tiny) = -708;
    # their mass sits ~48 sd left of the longshot, hence the wide referee
    mu_np = np.array([0.0, 0.0, 0.0, 0.0, 60.0])
    for k in (1, 2, 3):
        lp = float(wt.topk_log_prob(_t(mu_np), k)[-1])
        ref = _fine_topk(mu_np, np.ones(5), 4, k, width=70.0)
        assert ref < -1000.0
        assert abs(lp / ref - 1.0) < 1e-9, (k, lp, ref)


def test_863_ordinary_topk_equals_the_linear_count_program():
    # where nothing underflows the old linear program is exact; the log
    # domain must agree with it to rounding on ordinary fields
    from winning.torch.losses import _setup
    rng = np.random.default_rng(863)
    for n, spread, k in ((5, 1.0, 2), (12, 3.0, 3), (20, 2.0, 2)):
        mu = _t(rng.normal(size=n) * spread)
        sd = _t(rng.uniform(0.5, 2.0, size=n))
        _, _, _, mask, _, _, x, z, logf, logS, logZ = _setup(
            mu, sd, None, "normal", 257, 1e-12)
        S = torch.exp(logS)[0]
        F = torch.exp(torch.special.log_ndtr(z))[0]
        ref = []
        for i in range(n):
            pmf = torch.zeros(x.shape[-1], k, dtype=F64)
            pmf[:, 0] = 1.0
            for j in range(n):
                if j != i:
                    sh = torch.cat([torch.zeros_like(pmf[:, :1]),
                                    pmf[:, :-1]], dim=-1)
                    pmf = pmf * S[j, :, None] + sh * F[j, :, None]
            ref.append(torch.logsumexp(logf[0, i] + torch.log(pmf.sum(-1)),
                                       -1) - logZ[0])
        torch.testing.assert_close(wt.topk_log_prob(mu, k, sd),
                                   torch.stack(ref), rtol=1e-13, atol=1e-13)


# ------------------------------------- #864: analytic backward is once-only
def _hess(mode, checkpoint=False):
    m = _t([0.0, 1.0], grad=True)
    x = wt.race_lattice(m.detach())
    return torch.autograd.functional.hessian(
        lambda v: -wt.race_log_prob(v, lattice=x, backward=mode,
                                    checkpoint=checkpoint)[1], m)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_864_analytic_double_backward_raises(checkpoint):
    # it used to return a wrong, asymmetric Hessian silently
    with pytest.raises(RuntimeError, match="first-order only"):
        _hess("analytic", checkpoint)


def test_864_callers_of_the_analytic_backward_refuse_too():
    mu = _t([0.0, 0.5, 1.0], grad=True)
    V = _t([[0.3], [0.1], [-0.2]])
    for f in (lambda m: wt.thurstone_nll(m, torch.tensor(1)),
              lambda m: wt.race_log_prob_factor(m, V)[1]):
        with pytest.raises(RuntimeError, match="first-order only"):
            torch.autograd.grad(f(mu), mu, create_graph=True)
        # and plain first-order use is unaffected
        (g,) = torch.autograd.grad(f(mu), mu)
        assert torch.all(torch.isfinite(g))


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_864_autograd_backward_is_twice_differentiable(base):
    mu = _t([0.0, 0.7, 1.5], grad=True)
    sd = _t([1.0, 0.8, 1.3], grad=True)
    x = wt.race_lattice(mu.detach(), sd.detach(), base=base)
    f = lambda m, s: wt.race_log_prob(m, s, base=base, lattice=x,
                                      backward="autograd")
    assert torch.autograd.gradgradcheck(f, (mu, sd), eps=1e-6, atol=1e-6)
    H = _hess("autograd")
    torch.testing.assert_close(H, H.T, rtol=0, atol=1e-12)
    # the pair Hessian of -log Phi((mu0 - mu1)/sqrt 2): [[a, -a], [-a, a]]
    # (to lattice accuracy: the fixed lattice breaks translation exactly)
    assert abs(float(H[0, 0] + H[0, 1])) < 1e-9


def test_864_first_order_analytic_gradient_is_unchanged():
    mu = _t([0.0, 1.0, 2.5], grad=True)
    sd = _t([1.0, 0.7, 1.4], grad=True)
    ga = torch.autograd.grad(wt.race_log_prob(mu, sd)[1], (mu, sd))
    gb = torch.autograd.grad(wt.race_log_prob(mu, sd, backward="autograd")[1],
                             (mu, sd))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


# ------------------------------------------ #865: the lattice is scale-free
@pytest.mark.parametrize("scale", [1e-15, 1e-8, 1e8, 1e15])
def test_865_race_prefix_topk_are_invariant_under_rescaling(scale):
    # an absolute 1e-12 floor on the window width repriced [0, 1] with
    # sd 1 from [0.760, 0.240] to [0.681, 0.319] at scale 1e-15
    rng = np.random.default_rng(865)
    fields = [(np.array([0.0, 1.0]), np.ones(2)),
              (rng.normal(size=6) * 2.0, rng.uniform(0.5, 2.0, size=6)),
              (np.array([0.0, 0.0, 0.0, 0.0, 20.0]), np.ones(5))]

    def heads(mu, sd):
        return [wt.race_log_prob(mu, sd),
                wt.race_log_prob(mu, sd, base="gumbel"),
                wt.ordered_prefix_log_prob(mu, torch.tensor([1, 0]), sd),
                wt.ordered_prefix_log_prob(mu, torch.tensor([1, 0, 2])
                                           [:min(3, len(mu))], sd),
                wt.topk_log_prob(mu, 1, sd),
                wt.topk_log_prob(mu, 2, sd)]

    for mu, sd in fields:
        ref = heads(_t(mu), _t(sd))
        got = heads(_t(mu * scale), _t(sd * scale))
        for g, r in zip(got, ref):
            err = float(((g - r).abs() / r.abs().clamp(min=1.0)).max())
            assert err < 1e-12, (scale, mu, err)


def test_865_ordinary_lattices_never_reach_the_width_guard():
    # the guard is hi >= lo + max sd; on ordinary fields the window is
    # many sd wide, so neither the old floor nor the new one binds and
    # the lattice is bitwise what it was
    rng = np.random.default_rng(8650)
    for n in (1, 2, 3, 8, 20):
        for spread in (0.5, 2.0, 6.0):
            mu = rng.normal(size=n) * spread
            sd = rng.uniform(0.5, 2.0, size=n)
            for base in ("normal", "gumbel"):
                x = wt.race_lattice(_t(mu), _t(sd), base=base)
                assert float(x[-1] - x[0]) > 5.0 * sd.max(), (n, spread)
