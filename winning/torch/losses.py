"""Thurstonian losses for network heads: winner, ordered prefix, top-k.

All min-wins (lower mu is better), each normalised by the winner total
on its own lattice. k = 1 uses race_log_prob's lattice, so it reproduces
race_log_prob. Events with k >= 2 use a lattice that also covers the
later places, because their mass lies past the winner window (#753).

    thurstone_nll            analytic backward (race_log_prob)
    ordered_prefix_log_prob  AUTOGRAD backward through the lattice
    topk_log_prob            AUTOGRAD backward through a count program

The last two differentiate their forward with autograd, so they keep a
tape of O(k n L) (prefix) or O(n L k) (top-k) entries; analytic backwards
for them are future work (docs/design/torch.md).
"""

from __future__ import annotations

import math

import torch

from .race import (_as_base, _lattice_rows, _prepare, _terms,
                   _win_integrand, race_log_prob)

__all__ = ["thurstone_nll", "ordered_prefix_log_prob", "topk_log_prob"]


def _reduce(x, reduction):
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f'reduction must be "mean", "sum" or "none"; '
                     f'got {reduction!r}')


def thurstone_nll(mu, winner, sd=None, *, V=None, nodes=15, base="normal",
                  points=257, mask=None, reduction="mean"):
    """-log P(observed winner), the Thurstone replacement for cross-entropy.

    mu: (..., n) locations (min-wins: pass -scores for a max-wins head).
    winner: (...) long tensor of winning indices. V: optional (..., n,
    rank) loadings, for the factor race (race_log_prob_factor)."""
    if V is None:
        lp = race_log_prob(mu, sd, base=base, points=points, mask=mask)
    else:
        from .factor import race_log_prob_factor
        lp = race_log_prob_factor(mu, V, sd, nodes=nodes, base=base,
                                  points=points, mask=mask)
    winner = torch.as_tensor(winner, dtype=torch.long, device=mu.device)
    lw = torch.gather(lp, -1, winner.unsqueeze(-1).expand(
        *lp.shape[:-1], 1)).squeeze(-1)
    return _reduce(-lw, reduction)


def _setup(mu, sd, mask, base, points, delta, later=False):
    # later=True: the event involves later places, whose mass lies beyond
    # the winner window's right edge (#753: P(0 then 1) at a gap of 10
    # came back 0.503 for 1 - 8e-13); k = 1 keeps the winner window, so
    # it reproduces race_log_prob exactly
    fam = _as_base(base)
    mu_r, sd_r, mask_r, lead, n = _prepare(mu, sd, mask)
    x = _lattice_rows(mu_r.detach(), sd_r.detach(), mask_r, fam, points,
                      delta, cover_all=later)
    z, logf, logS = _terms(mu_r, sd_r, mask_r, x, fam)
    g = _win_integrand(logf, logS)
    g = torch.where(mask_r[:, :, None], g, torch.full_like(g, -math.inf))
    logZ = torch.logsumexp(g.flatten(1), dim=-1)                  # (R,)
    return fam, mu_r, sd_r, mask_r, lead, n, x, z, logf, logS, logZ


def _cum_integral_log(h, dh, log_dx):
    """log of the running integral int_{-inf}^{x_l} exp(h), per row.

    Cumulative trapezoid plus the Euler-Maclaurin end correction
    -dx^2/12 h'(x_l) (h' at the far-left start is negligible), so the
    error is fourth order in the spacing rather than second. dh is
    d log h / dx. The first entry stands in for the (negligible) mass
    below the lattice with a half-cell; the correction is clamped to
    +-50% there, where the running integral is a few cells wide."""
    a = torch.logaddexp(h[:, :-1], h[:, 1:]) + (log_dx - math.log(2.0))
    first = h[:, :1] + (log_dx - math.log(2.0))
    C = torch.logcumsumexp(torch.cat([first, a], dim=-1), dim=-1)
    corr = torch.exp(2.0 * log_dx + h - C) * dh / 12.0
    return C + torch.log1p(-corr.clamp(-0.5, 0.5))


def ordered_prefix_log_prob(mu, order, sd=None, *, base="normal",
                            points=257, delta=1e-12, mask=None):
    """log P(order[..., 0] first, order[..., 1] second, ...), min-wins.

    order: (..., k) long tensor of distinct valid runner indices; k = 1 is
    race_log_prob at the winner. The full permutation likelihood is the
    k = n case, but the cost and accuracy argument below is for short
    prefixes (k = 2, 3), which is what restriction tests need.

        P = int f_{o_k}(x) H_{k-1}(x) prod_{l not in prefix} S_l(x) dx
        H_1 = F_{o_1},  H_m(x) = int_{-inf}^x f_{o_m} H_{m-1}

    k <= 2 is spectrally accurate on the lattice (H_1 is the exact cdf);
    each further level adds a cumulative trapezoid with an Euler-Maclaurin
    end correction, fourth order in the spacing (docs/design/torch.md
    gives measured errors; raise points= for k >= 3). Backward: autograd
    through the lattice.
    """
    order = torch.as_tensor(order, dtype=torch.long, device=mu.device)
    k = order.shape[-1]
    fam, mu_r, sd_r, mask_r, lead, n, x, z, logf, logS, logZ = _setup(
        mu, sd, mask, base, points, delta, later=k > 1)
    if k < 1 or k > n:
        raise ValueError(f"order must hold 1..n indices; got k = {k}")
    o = order.expand(*lead, k).reshape(-1, k)
    if k > 1 and not torch.jit.is_tracing():
        srt = torch.sort(o, dim=-1).values
        if bool((srt[:, 1:] == srt[:, :-1]).any()):
            raise ValueError("order has a repeated runner")
    L = x.shape[-1]
    idx = o[:, :, None].expand(-1, k, L)
    lf = torch.gather(logf, 1, idx)                               # (R, k, L)
    lS = torch.gather(logS, 1, idx)
    rest = logS.sum(1) - lS.sum(1)                                # (R, L)
    if k == 1:
        integrand = lf[:, 0] + rest
    else:
        zo = torch.gather(z, 1, idx)                              # (R, k, L)
        so = torch.gather(sd_r, 1, o)[:, :, None]                 # (R, k, 1)
        H = fam.log_cdf(zo[:, 0])                                 # (R, L)
        dH = torch.exp(lf[:, 0] - H)                              # d H / dx
        log_dx = torch.log(x[:, 1:2] - x[:, :1])
        for m in range(1, k - 1):
            h = lf[:, m] + H
            dh = fam.dlog_pdf(zo[:, m]) / so[:, m] + dH
            H = _cum_integral_log(h, dh, log_dx)
            dH = torch.exp(h - H)
        integrand = lf[:, k - 1] + H + rest
    out = torch.logsumexp(integrand, dim=-1) - logZ
    return out.reshape(lead)


def topk_log_prob(mu, k, sd=None, *, base="normal", points=257,
                  delta=1e-12, mask=None):
    """log P(runner i finishes among the k smallest), every i, min-wins.

        P_i = int f_i(x) P(#{j != i : X_j < x} <= k - 1) prod... dx

    The count law of the OTHER runners at each lattice point comes from
    prefix and suffix Poisson-binomial programs (truncated at k counts),
    joined per runner -- no deconvolution, so no cancellation. Cost
    O(n L k^2), with a Python loop over runners (2n steps) and an
    autograd tape of O(n L k); float64 recommended for k > 1. The sum
    over valid runners is k (to lattice accuracy).
    """
    if not (isinstance(k, int) and k >= 1):
        raise ValueError(f"k must be a positive int; got {k!r}")
    fam, mu_r, sd_r, mask_r, lead, n, x, z, logf, logS, logZ = _setup(
        mu, sd, mask, base, points, delta, later=k > 1)
    R, _, L = z.shape
    m3 = mask_r[:, :, None]
    S = torch.where(m3, torch.exp(logS), torch.ones_like(logS))
    F = torch.where(m3, torch.exp(fam.log_cdf(z)), torch.zeros_like(logS))
    zero = torch.zeros(R, L, 1, dtype=z.dtype, device=z.device)

    def step(pmf, j):
        # add runner j to a count pmf (R, L, k), truncated at k - 1
        shifted = torch.cat([zero, pmf[..., :-1]], dim=-1)
        return pmf * S[:, j, :, None] + shifted * F[:, j, :, None]

    start = torch.zeros(R, L, k, dtype=z.dtype, device=z.device)
    start[..., 0] = 1.0
    pre = [start]
    for j in range(n):
        pre.append(step(pre[-1], j))
    suf = [start]
    for j in range(n - 1, -1, -1):
        suf.append(step(suf[-1], j))
    suf = suf[::-1]                         # suf[j]: runners j..n-1
    out = []
    for i in range(n):
        # P(count of others <= k-1) = sum_a pre_i[a] * cdf(suf_{i+1})[k-1-a]
        cdf = torch.cumsum(suf[i + 1], dim=-1)
        le = (pre[i] * torch.flip(cdf, dims=[-1])).sum(-1)       # (R, L)
        integrand = logf[:, i] + torch.log(le.clamp(min=torch.finfo(
            z.dtype).tiny))
        out.append(torch.logsumexp(integrand, dim=-1))
    lp = torch.stack(out, dim=-1) - logZ[:, None]
    lp = torch.where(mask_r, lp, torch.full_like(lp, -math.inf))
    return lp.reshape(*lead, n)
