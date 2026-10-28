"""The factor (correlated) Thurstone race as a fixed Gauss-Hermite mixture.

X_i = mu_i + V_i . f + sd_i Z_i with f ~ N(0, I_rank). Conditional on f
the race is independent with locations mu + V f, so

    log P(i wins) = logsumexp_q [ log w_q + race_log_prob(mu + V f_q, sd)_i ]

over a fixed probabilists' Gauss-Hermite tensor rule (nodes per dimension,
rank 1 or 2 in practice: nodes**rank conditional races per row). The node
axis is just another batch axis for race_log_prob, so every conditional
race uses the analytic O(n L) backward; autograd only sees the cheap
(nodes x n) affine map mu + V f_q and the final logsumexp over nodes.

Unlike winning.factor.race_probabilities there is no adaptive node rule
(the order is fixed so shapes are static); a field whose loadings are
large against sd -- the near-deterministic conditional race -- needs more
nodes than the default, as docs/design/torch.md says.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .race import race_log_prob

__all__ = ["race_log_prob_factor", "hermite_rule"]


def hermite_rule(rank, nodes, dtype=torch.float64, device=None):
    """Tensor probabilists' Gauss-Hermite rule for N(0, I_rank).

    Returns (F, log_w): F (nodes**rank, rank), log weights summing to 0
    in exp."""
    if not (isinstance(nodes, int) and nodes >= 1):
        raise ValueError(f"nodes must be a positive int; got {nodes!r}")
    t, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    grids = np.meshgrid(*([t] * rank), indexing="ij")
    F = np.stack([g.ravel() for g in grids], axis=1)
    wg = np.meshgrid(*([w] * rank), indexing="ij")
    W = np.prod(np.stack([g.ravel() for g in wg], axis=1), axis=1)
    return (torch.as_tensor(F, dtype=dtype, device=device),
            torch.as_tensor(np.log(W), dtype=dtype, device=device))


def race_log_prob_factor(mu, V, sd=None, *, nodes=15, base="normal",
                         points=257, delta=1e-12, mask=None,
                         backward="analytic"):
    """log P(runner i wins) under factor loadings V, batched, min-wins.

    mu: (..., n). V: (..., n, rank) torch tensor -- one row per runner,
    exactly that shape (no spelling normalisation: with leading batch
    dims a (rank, n) matrix is ambiguous). sd: idiosyncratic sd, as in
    race_log_prob. nodes: Gauss-Hermite order per factor dimension.
    Gradients flow to mu, V and sd. backward= is race_log_prob's: the
    default "analytic" is first-order only (double backward raises,
    #864); pass backward="autograd" for second derivatives.
    """
    if not torch.is_tensor(V):
        raise TypeError("V must be a torch tensor of shape (..., n, rank)")
    if V.dim() < 2 or V.shape[-2] != mu.shape[-1]:
        raise ValueError(
            f"V must have shape (..., n, rank) with n = {mu.shape[-1]}; "
            f"got {tuple(V.shape)}")
    rank = V.shape[-1]
    F, log_w = hermite_rule(rank, nodes, dtype=mu.dtype, device=mu.device)
    # (..., Q, n): every node's conditional locations
    shift = torch.einsum("...nr,qr->...qn", V, F)
    mu_q = mu.unsqueeze(-2) + shift
    sd_q = None
    if sd is not None:
        sd_q = torch.as_tensor(sd, dtype=mu.dtype, device=mu.device)
        if sd_q.dim() >= 1:
            sd_q = sd_q.unsqueeze(-2)          # (..., 1, n): every node
    mask_q = None if mask is None else torch.as_tensor(
        mask, dtype=torch.bool, device=mu.device).unsqueeze(-2)
    lp = race_log_prob(mu_q, sd_q, base=base, points=points, delta=delta,
                       mask=mask_q, backward=backward)
    if mask is None:
        return torch.logsumexp(lp + log_w[:, None], dim=-2)
    # padded runners carry -inf at every node: swap in a finite stand-in
    # before the node logsumexp (whose backward would otherwise meet
    # exp(-inf - -inf)), then put -inf back
    valid = mask_q.expand(lp.shape)
    lp = torch.where(valid, lp, torch.zeros_like(lp))
    out = torch.logsumexp(lp + log_w[:, None], dim=-2)
    return torch.where(valid[..., 0, :], out,
                       torch.full_like(out, -math.inf))
