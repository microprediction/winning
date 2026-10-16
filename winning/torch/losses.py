"""Thurstonian losses for network heads: winner, ordered prefix, top-k.

All min-wins (lower mu is better). The prefix and top-k losses share
one EVENT lattice (_event_lattice_rows): race_log_prob's window widened
to cover every runner's bulk, since a later place lives where a winner
does not (#753). Each is normalised by the win-integrand total on that
same lattice, so k = 1 of each reproduces race_log_prob to lattice
accuracy (~1e-12) and the families are mutually consistent.

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

from .race import (_as_base, _prepare, _terms, _win_integrand,
                   _window_rows, race_log_prob)

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


# The prefix / top-k lattice: spacing at most this fraction of the
# smallest valid sd, and at most this many points (or points=, if more).
_EVENT_SPACING = 0.3
_EVENT_CAP = 4097
_EVENT_WARN_SPACING = 1.0      # x min sd; see _event_lattice_rows


@torch.no_grad()
def _event_lattice_rows(mu, sd, mask, fam, points, delta, nested=False):
    """The lattice for ordered-prefix and top-k events (#753).

    The win lattice (race._window_rows) ends where every WINNER integrand
    is negligible. A later place, or a top-k membership, can have its
    mass far to the right of that (P(0 then 1) at mu = [0, 10] came out
    0.503 on it). Here the right edge also covers every valid runner's
    bulk, measured in the WIDEST valid sd:

        hi = max(hi_win, max_i mu_i + q_hi max_i sd_i),   S(q_hi) = delta.

    Why the widest sd: the last runner o_k of a prefix must finish behind
    the others, so the integrand f_{o_k} H rest has its mass at or left
    of about max_i mu_i, often deep in o_k's own right tail (P(1 then 0)
    at mu = [0, 5], sd = [1, 0.3] is 8e-7, all of it near x = 4.6). For
    a log-concave base, S(z + q) / S(z) <= S(q) / S(0) for z >= 0, so
    with x0 = max_i mu_i >= mu_{o_k} and a step of q_hi max sd >= q_hi
    sd_{o_k}, what is dropped right of hi is ~delta RELATIVE to the mass
    near x0, not merely delta absolute (mu_i + q_hi sd_i per runner left
    7e-7 relative error on that event). The left edge is the win
    lattice's, which already covers min_i mu_i + sd_i q_lo relative to
    each runner's win share.

    The point count is the larger of points= and what keeps the spacing
    at most _EVENT_SPACING * min sd (enough for the spectral k <= 2 and
    top-k integrals), up to _EVENT_CAP (or points=, if that is larger).
    nested=True (prefixes with k >= 3, whose cumulative integrals are
    fourth order, so their error follows the spacing) also keeps the
    spacing no coarser than the win lattice's at points=, as far as twice
    points= (without it, rare k = 3 orders of ordinary fields lost up to
    10x accuracy to the wider window). Past the cap the window still covers every event;
    only the spacing coarsens (accuracy: docs/design/torch.md). Reading
    the count is one host sync per call; it is the batch maximum, so all
    rows of a batch share it."""
    if mu.device.type == "mps":
        cpu = lambda t: None if t is None else t.cpu()
        return _event_lattice_rows(cpu(mu), cpu(sd), cpu(mask), fam,
                                   points, delta, nested).to(mu.device)
    wd = torch.float64
    lo, hi_win = _window_rows(mu, sd, mask, fam, delta)
    if mask is None:
        mask = torch.ones(mu.shape, dtype=torch.bool, device=mu.device)
    m64 = mu.to(wd)
    s64 = sd.to(wd)
    q_hi = fam.upper_quantile(torch.tensor(delta, dtype=wd,
                                           device=mu.device))
    mmax = torch.where(mask, m64, torch.full_like(m64, -math.inf))
    smax = torch.where(mask, s64, torch.zeros_like(s64))
    hi = torch.maximum(hi_win,
                       mmax.max(-1).values + q_hi * smax.max(-1).values)
    smin = torch.where(mask, s64, torch.full_like(s64, math.inf))
    cap = max(points, _EVENT_CAP)

    def need(cells):
        # host sync: the lattice length is the batch maximum
        w = float(torch.nan_to_num(cells, nan=0.0, posinf=cap).max()) \
            if cells.numel() else 0.0
        return math.ceil(w) + 1

    L = max(points, need((hi - lo) / (_EVENT_SPACING * smin.min(-1).values)))
    if nested:
        # no coarser than the win lattice at points=, up to 2x points
        L = max(L, min(need((hi - lo) / (hi_win - lo) * (points - 1)),
                       2 * points - 1))
    # what the cap leaves: accuracy holds to a spacing of ~1.25 min sd
    # (equal-mean pair, sd = [1, r]: error 1e-12 at 0.83, 5e-9 at 1.25,
    # 1e-3 at 2.1, 0.16 at 4.2 -- top-2 1.16 at r = 1000, #756). Past
    # 1.0 min sd, say so rather than refine without bound; points= above
    # the cap is honoured.
    worst = (hi - lo) / ((cap - 1) * smin.min(-1).values)
    worst = float(torch.nan_to_num(worst, nan=0.0, posinf=0.0).max()) \
        if worst.numel() else 0.0
    if worst > _EVENT_WARN_SPACING:
        import warnings
        warnings.warn(
            "winning.torch event lattice cannot resolve the narrowest "
            f"runner: at its cap of {cap} points the spacing is "
            f"{worst:.2g} x the smallest sd (accurate to ~1.25). The "
            "runners' scales differ too much for one uniform lattice, so "
            "ordered-prefix and top-k probabilities may be inaccurate. "
            f"Pass points={math.ceil(worst * (cap - 1)) + 1} or more.",
            RuntimeWarning, stacklevel=4)
    L = int(min(L, cap))
    u = torch.linspace(0.0, 1.0, L, dtype=wd, device=mu.device)
    x = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    return x.to(mu.dtype)


def _setup(mu, sd, mask, base, points, delta, nested=False):
    fam = _as_base(base)
    mu_r, sd_r, mask_r, lead, n = _prepare(mu, sd, mask)
    x = _event_lattice_rows(mu_r.detach(), sd_r.detach(), mask_r, fam,
                            points, delta, nested)
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
        mu, sd, mask, base, points, delta, nested=k >= 3)
    if k < 1 or k > n:
        raise ValueError(f"order must hold 1..n indices; got k = {k}")
    o = order.expand(*lead, k).reshape(-1, k)
    if not torch.jit.is_tracing():
        if bool(((o < 0) | (o >= n)).any()):
            raise ValueError(f"order has an index outside 0..{n - 1}")
        if k > 1:
            srt = torch.sort(o, dim=-1).values
            if bool((srt[:, 1:] == srt[:, :-1]).any()):
                raise ValueError("order has a repeated runner")
        # a masked runner is padding, not a finisher: gathering it priced
        # a "probability" of 7.1 and sent gradient to padding (#760)
        if not bool(torch.gather(mask_r, 1, o).all()):
            raise ValueError("order selects a masked (padding) runner")
    L = x.shape[-1]
    idx = o[:, :, None].expand(-1, k, L)
    lf = torch.gather(logf, 1, idx)                               # (R, k, L)
    # log S of the runners NOT in the prefix, summed directly. The
    # difference sum(all) - sum(prefix) cancels where a prefix runner's
    # log S is huge, which the wider event lattice reaches: gumbel log S
    # is -e^u, ~-1e17 forty sd out, so the difference lost every digit
    # (#753). Here nothing is subtracted.
    inpre = torch.zeros(logS.shape[:2], dtype=torch.bool,
                        device=logS.device).scatter(1, o, True)
    rest = torch.where(inpre[:, :, None], torch.zeros_like(logS),
                       logS).sum(1)                               # (R, L)
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
        mu, sd, mask, base, points, delta)
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
