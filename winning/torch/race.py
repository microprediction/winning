"""The independent Thurstone race in PyTorch, with an analytic backward.

Min-wins, as everywhere in winning: runner i performs X_i = mu_i + sd_i Z_i
with Z_i a standardized base (normal, or the unit-variance min-Gumbel that
makes the race exactly softmax), and the SMALLEST X_i wins. A network that
emits utilities (higher is better) passes mu = -utility.

    log p_i = A_i - logsumexp_j A_j,
    A_i     = logsumexp_x [ log f_i(x) + sum_{j != i} log S_j(x) ]

over a uniform lattice x of `points` values per row. The rectangle weight
dx is common to every runner and cancels in the normalisation, exactly as
in winning.factor.race_probabilities, which also divides by the lattice
total. Everything is in the log domain, so a runner 10 or 40 sd behind
has a finite log-probability in float32.

The backward is the vector-Jacobian product computed from the same
per-lattice arrays (see docs/design/torch.md): with upstream gradient G,

    c_i       = G_i - p_i sum_j G_j
    r_i(x)    = c_i softmax_x(g_i)(x)          (responsibilities, n x L)
    dL/dlog f_k(x) = r_k(x)
    dL/dlog S_k(x) = sum_{i != k} r_i(x) = R(x) - r_k(x)

so one O(n L) pass gives the gradient with respect to every mu_k and sd_k.
No autograd graph over n x L (or n(n-1) pair) terms is kept; the backward
recomputes the lattice terms from (mu, sd, x) instead of saving them, so
peak memory is a few n x L buffers per row, not a tape.

The lattice is a function of (mu, sd) but is treated as a constant in the
backward (it is computed under no_grad). The exact win integral does not
depend on where the lattice sits, so the omitted term is the derivative of
the discretisation error, which is below the forward's own accuracy.
"""

from __future__ import annotations

import math

import torch

__all__ = ["race_log_prob", "race_lattice", "BASES"]

_EULER = 0.5772156649015329
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_GUMBEL_C = math.pi / math.sqrt(6.0)


# ----------------------------------------------------------------- bases
_SQRT1_2 = math.sqrt(0.5)


def _log_ndtr(z):
    """log Phi(z). torch.special.log_ndtr where the device has it; MPS
    (torch 2.14) does not, so there it is assembled from erfcx/erfc:
    log(erfcx(t)/2) - t^2 with t = -z/sqrt 2 below z = -1 (no underflow
    however far out), log(erfc(t)/2) on [-1, 0], log1p(-erfc(-t)/2)
    above. Each branch sees a clamped argument so the unused ones stay
    finite under autograd."""
    if z.device.type != "mps":
        return torch.special.log_ndtr(z)
    t = -z * _SQRT1_2
    t_lo = torch.clamp(t, min=_SQRT1_2)              # z <= -1 branch
    lo = torch.log(0.5 * torch.special.erfcx(t_lo)) - t_lo * t_lo
    t_mid = torch.clamp(t, min=0.0, max=_SQRT1_2)
    mid = torch.log(0.5 * torch.erfc(t_mid))
    t_hi = torch.clamp(t, max=0.0)
    hi = torch.log1p(-0.5 * torch.erfc(-t_hi))
    return torch.where(z <= -1.0, lo, torch.where(z <= 0.0, mid, hi))


# Each base gives, for standardized z: log f, log S (survival), and their
# z-derivatives, plus log F (lower tail) for the order/membership losses.
# Normal tails go through log_ndtr, which is accurate far into both tails
# in float32 and float64 (no 1 - ndtr cancellation, cf. winning #96).

class _Normal:
    name = "normal"

    @staticmethod
    def log_pdf(z):
        return -0.5 * z * z - _LOG_SQRT_2PI

    @staticmethod
    def log_sf(z):
        return _log_ndtr(-z)

    @staticmethod
    def log_cdf(z):
        return _log_ndtr(z)

    @staticmethod
    def dlog_pdf(z):
        return -z

    @classmethod
    def dlog_sf(cls, z, logf0=None, logS=None):
        # minus the hazard, from logs: exact where S underflows. The
        # backward passes the log density and log survival it already
        # holds, so the tail function is not evaluated again.
        if logf0 is None:
            logf0 = cls.log_pdf(z)
        if logS is None:
            logS = cls.log_sf(z)
        return -torch.exp(logf0 - logS)

    @staticmethod
    def lower_quantile(p):
        # z with F(z) = p (p a float64 tensor)
        return torch.special.ndtri(p)

    @staticmethod
    def upper_quantile(p):
        # z with S(z) = p
        return -torch.special.ndtri(p)


class _Gumbel:
    """Standardized min-Gumbel: S(z) = exp(-exp(c z - gamma)), c = pi/sqrt 6.

    The same law as winning's base="gumbel"; the race over it is exactly
    softmax(-c mu / sd) for a common sd."""
    name = "gumbel"
    # exp(u) is clamped so S never reaches exp(-inf); far outside any window
    _UMAX = 60.0

    @classmethod
    def _u(cls, z):
        return torch.clamp(_GUMBEL_C * z - _EULER, max=cls._UMAX)

    @classmethod
    def log_pdf(cls, z):
        u = cls._u(z)
        return math.log(_GUMBEL_C) + u - torch.exp(u)

    @classmethod
    def log_sf(cls, z):
        return -torch.exp(cls._u(z))

    @classmethod
    def log_cdf(cls, z):
        # log(1 - exp(-e^u)), stable both ways
        e = torch.exp(cls._u(z))
        return torch.where(e < 0.6931, torch.log(-torch.expm1(-e)),
                           torch.log1p(-torch.exp(-e)))

    @classmethod
    def dlog_pdf(cls, z):
        return _GUMBEL_C * (1.0 - torch.exp(cls._u(z)))

    @classmethod
    def dlog_sf(cls, z, logf0=None, logS=None):
        return -_GUMBEL_C * torch.exp(cls._u(z))

    @staticmethod
    def lower_quantile(p):
        # F(z) = 1 - exp(-e^u) = p  ->  u = log(-log1p(-p))
        return (torch.log(-torch.log1p(-p)) + _EULER) / _GUMBEL_C

    @staticmethod
    def upper_quantile(p):
        # S(z) = p -> e^u = -log p
        return (torch.log(-torch.log(p)) + _EULER) / _GUMBEL_C


BASES = {"normal": _Normal, "gumbel": _Gumbel}


def _as_base(base):
    if isinstance(base, str) and base in BASES:
        return BASES[base]
    raise ValueError(f"base must be one of {sorted(BASES)}; got {base!r}")


# ---------------------------------------------------------------- inputs
def _prepare(mu, sd, mask):
    """Broadcast to (..., n), flatten leading dims to rows: (R, n)."""
    if not torch.is_tensor(mu):
        raise TypeError("mu must be a torch tensor of shape (..., n)")
    if mu.dim() == 0:
        raise ValueError("mu must have at least one dimension (runners)")
    if not mu.is_floating_point():
        raise TypeError("mu must be a floating-point tensor")
    if sd is None:
        sd = torch.ones((), dtype=mu.dtype, device=mu.device)
    elif not torch.is_tensor(sd):
        sd = torch.as_tensor(sd, dtype=mu.dtype, device=mu.device)
    shape = torch.broadcast_shapes(mu.shape, sd.shape)
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=mu.device)
        shape = torch.broadcast_shapes(shape, mask.shape)
    n = shape[-1]
    lead = shape[:-1]
    mu_r = mu.expand(shape).reshape(-1, n)
    sd_r = sd.expand(shape).reshape(-1, n)
    if mask is None:
        mask_r = torch.ones(mu_r.shape, dtype=torch.bool, device=mu.device)
    else:
        mask_r = mask.expand(shape).reshape(-1, n)
    return mu_r, sd_r, mask_r, lead, n


def race_lattice(mu, sd=None, *, base="normal", points=257, delta=1e-12,
                 mask=None):
    """The deterministic, vectorised lattice used by race_log_prob.

    Returns x of shape (..., points). Per row (see docs/design/torch.md):

      lo   = min_i mu_i + sd_i q_lo,      F(q_lo) = delta / n_valid
      hi_i = min(mu_i + sd_i q_hi,  m_i + q_hi^N s_i),   S(q_hi) = delta
      hi   = max_i hi_i

    where (m_i, s_i) is the Gaussian product of runner i with the best
    other runner k (lowest mu): the integrand f_i prod_{j != i} S_j is
    f_i S_k times a non-increasing function, so its normalised right tail
    is dominated by that of f_i S_k. No search, no data-dependent
    control flow: one pass of min/max over the runners.
    """
    fam = _as_base(base)
    mu_r, sd_r, mask_r, lead, n = _prepare(mu, sd, mask)
    x = _lattice_rows(mu_r.detach(), sd_r.detach(), mask_r, fam, points,
                      delta)
    return x.reshape(*lead, points)


@torch.no_grad()
def _lattice_rows(mu, sd, mask, fam, points, delta, cover_all=False):
    """The per-row lattice. cover_all=False is the WINNER window: its
    right edge is cut where every winner integrand f_i prod S is
    negligible. Later-place events (an ordered prefix's second finisher,
    top-k membership) have their mass around the later runners' own
    bodies, beyond that cut, so they pass cover_all=True: the right edge
    then covers every valid runner to its own upper quantile (#753)."""
    if mu.device.type == "mps":
        # MPS has neither float64 nor ndtri; the window is O(17 n) per
        # row, so it is computed on the CPU and only the lattice moves
        cpu = lambda t: None if t is None else t.cpu()
        return _lattice_rows(cpu(mu), cpu(sd), cpu(mask), fam, points,
                             delta, cover_all).to(mu.device)
    dt = mu.dtype
    wd = torch.float64
    big = torch.finfo(wd).max
    if mask is None:
        mask = torch.ones(mu.shape, dtype=torch.bool, device=mu.device)
    m64 = mu.to(wd)
    s64 = sd.to(wd)
    dlt = torch.tensor(delta, dtype=wd, device=mu.device)
    nval = mask.sum(-1, keepdim=True).clamp(min=1).to(wd)
    q_lo = fam.lower_quantile(dlt / nval)                        # (R, 1)
    q_hi = fam.upper_quantile(dlt)
    qn = -torch.special.ndtri(dlt)
    lo_i = torch.where(mask, m64 + s64 * q_lo, torch.full_like(m64, big))
    lo = lo_i.min(-1).values
    # Right edge. Runner i's integrand is f_i S_k times a non-increasing
    # function for any other runner k, so its normalised right tail is
    # dominated by that of f_i S_k; with k the lowest-mu other runner,
    # the Gaussian product of the two gives the edge (and f_i alone
    # gives mu_i + q_hi sd_i, whichever is nearer).
    key = torch.where(mask, m64, torch.full_like(m64, math.inf))
    if cover_all:
        hi_i = m64 + s64 * q_hi
    elif mu.shape[-1] >= 2:
        two = torch.topk(key, 2, dim=-1, largest=False)
        k1, k2 = two.indices[..., :1], two.indices[..., 1:2]
        idx = torch.arange(mu.shape[-1], device=mu.device)[None, :]
        kk = torch.where(idx == k1, k2, k1)
        mk = torch.gather(m64, -1, kk)
        sk = torch.gather(s64, -1, kk)
        okk = torch.gather(mask, -1, kk)
        prec = 1.0 / s64 ** 2 + 1.0 / sk ** 2
        m_pair = (m64 / s64 ** 2 + mk / sk ** 2) / prec
        pair = m_pair + qn * prec.rsqrt()
        own = m64 + s64 * q_hi
        hi_i = torch.where(okk, torch.minimum(own, pair), own)
    else:
        hi_i = m64 + s64 * q_hi
    hi_i = torch.where(mask, hi_i, torch.full_like(m64, -big))
    hi = hi_i.max(-1).values
    # Left edge, RELATIVE to each runner's own share: the mass of runner
    # i's integrand left of lo is at most F_i(lo), which must be small
    # against p_i, not against 1. A noisy longshot behind a sharp
    # favourite otherwise loses ~1e-8 of its share. p_i is bounded below
    # by F_i(t) prod_{j != i} S_j(t) at any t (i under t, the rest over
    # it); 17 probes over the provisional window give that bound in
    # O(17 n), and lo moves left to F_i(lo) = delta * bound where needed.
    t = lo[:, None] + (hi - lo)[:, None] * torch.linspace(
        0.0, 1.0, 17, dtype=wd, device=mu.device)[None, :]
    zt = (t[:, None, :] - m64[:, :, None]) / s64[:, :, None]
    m3 = mask[:, :, None]
    lS = torch.where(m3, fam.log_sf(zt), torch.zeros_like(zt))
    lF = torch.where(m3, fam.log_cdf(zt), torch.zeros_like(zt))
    lb = (lF + lS.sum(1, keepdim=True) - lS).max(-1).values      # (R, n)
    floor = math.log(torch.finfo(wd).tiny) + 2.0
    target = torch.exp((lb + math.log(delta)).clamp(min=floor))
    lo_rel = torch.where(mask, m64 + s64 * fam.lower_quantile(target),
                         torch.full_like(m64, big))
    lo = torch.minimum(lo, lo_rel.min(-1).values)
    # a one-runner (or degenerate) row still needs a non-empty lattice
    smax = torch.where(mask, s64, torch.zeros_like(s64)).max(-1).values
    hi = torch.maximum(hi, lo + smax.clamp(min=1e-12))
    u = torch.linspace(0.0, 1.0, points, dtype=wd, device=mu.device)
    x = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    return x.to(dt)


# ------------------------------------------------------------- the kernel
def _terms(mu, sd, mask, x, fam):
    """z, log f (incl. -log sd), log S, with padding neutralised.

    mask None means no padding (and skips every where)."""
    z = (x[:, None, :] - mu[:, :, None]) / sd[:, :, None]
    logf = fam.log_pdf(z) - torch.log(sd)[:, :, None]
    logS = fam.log_sf(z)
    if mask is not None:
        # padding gets FINITE neutral values (log f = 0, log S = 0) and is
        # cut out of A below: -inf here would put 0 * nan into autograd's
        # logsumexp backward for a padded row
        m3 = mask[:, :, None]
        logf = torch.where(m3, logf, torch.zeros_like(z))
        logS = torch.where(m3, logS, torch.zeros_like(z))
    return z, logf, logS


def _win_integrand(logf, logS):
    """g_i(x) = log f_i(x) + sum_{j != i} log S_j(x)  (R, n, L)."""
    return logf + (logS.sum(1, keepdim=True) - logS)


def _normalise(g, mask):
    A = torch.logsumexp(g, dim=-1)                                # (R, n)
    if mask is not None:
        A = torch.where(mask, A, torch.full_like(A, -math.inf))
    return A - torch.logsumexp(A, dim=-1, keepdim=True), A


def _forward_rows(mu, sd, mask, x, fam):
    z, logf, logS = _terms(mu, sd, mask, x, fam)
    return _normalise(_win_integrand(logf, logS), mask) + (z, logf, logS)


class _RaceLogProb(torch.autograd.Function):
    """Forward as _forward_rows; backward the O(n L) VJP.

    Saves log S (one (R, n, L) tensor) so the backward never re-evaluates
    the expensive tail function; checkpoint=True saves nothing of that
    size and recomputes it instead (half the memory, ~1.5x the time)."""

    @staticmethod
    def forward(ctx, mu, sd, x, mask, fam, checkpoint):
        out, A, z, logf, logS = _forward_rows(mu, sd, mask, x, fam)
        ctx.fam = fam
        ctx.has_mask = mask is not None
        ctx.checkpoint = checkpoint
        keep = [mu, sd, x, out, A]
        if mask is not None:
            keep.append(mask)
        if not checkpoint:
            keep.append(logS)
        ctx.save_for_backward(*keep)
        return out

    @staticmethod
    def backward(ctx, G):
        saved = list(ctx.saved_tensors)
        mu, sd, x, out, A = saved[:5]
        rest_ = saved[5:]
        mask = rest_.pop(0) if ctx.has_mask else None
        fam = ctx.fam
        z = (x[:, None, :] - mu[:, :, None]) / sd[:, :, None]
        log_sd = torch.log(sd)[:, :, None]
        logf0 = fam.log_pdf(z)                       # standardized, no sd
        logS = fam.log_sf(z) if ctx.checkpoint else rest_.pop(0)
        if mask is not None:
            # padding: neutral values, as in _terms (its dz is zeroed)
            m3 = mask[:, :, None]
            logf0 = torch.where(m3, logf0, torch.zeros_like(z))
            log_sd = torch.where(m3, log_sd, torch.zeros_like(log_sd))
            logS = torch.where(m3, logS, torch.zeros_like(z))
        g = (logf0 - log_sd) + (logS.sum(1, keepdim=True) - logS)
        p = torch.exp(out)
        if mask is not None:
            G = torch.where(mask, G, torch.zeros_like(G))
        c = G - p * G.sum(-1, keepdim=True)                       # (R, n)
        if mask is not None:
            c = torch.where(mask, c, torch.zeros_like(c))
            A = torch.where(mask, A, torch.zeros_like(A))
        r = c[:, :, None] * torch.exp(g - A[:, :, None])         # (R, n, L)
        rest = r.sum(1, keepdim=True) - r                        # sum_{i!=k}
        dz = r * fam.dlog_pdf(z) + rest * fam.dlog_sf(z, logf0, logS)
        if mask is not None:
            dz = torch.where(mask[:, :, None], dz, torch.zeros_like(dz))
        inv = 1.0 / sd
        grad_mu = -dz.sum(-1) * inv
        grad_sd = -((dz * z).sum(-1) + c) * inv
        return grad_mu, grad_sd, None, None, None, None


def race_log_prob(mu, sd=None, *, base="normal", points=257, delta=1e-12,
                  mask=None, lattice=None, backward="analytic",
                  checkpoint=False):
    """log P(runner i wins), min-wins, for every runner, batched.

    mu:   (..., n) locations (lower is better; pass -utility for max-wins).
    sd:   None (unit), a scalar, or a tensor broadcastable to mu: each
          runner's performance sd. Gradients flow to mu and to sd.
    base: "normal" (Thurstone) or "gumbel" (standardized min-Gumbel; the
          race is then exactly log_softmax(-pi mu / (sqrt(6) sd)) for a
          common sd -- a built-in control).
    points: lattice points per row (fixed, so shapes are static).
    delta:  tail mass the window may drop (relative, per runner; see
          race_lattice and docs/design/torch.md).
    mask: optional bool (..., n); False marks padding. Padding runners do
          not run: they get log-prob -inf and zero gradient, and the
          valid runners' results equal the unpadded race's.
    lattice: optional precomputed (..., points) lattice (e.g. from
          race_lattice) -- fixed, so finite differences see the same grid.
    backward: "analytic" (default; the O(n L) vector-Jacobian product)
          or "autograd" (differentiate the forward with autograd; for
          testing and timing comparison only).
    checkpoint: analytic backward only. False (default) saves log S, one
          (R, n, L) tensor, for the backward; True saves nothing that
          size and recomputes it (less memory, more time).

    Returns (..., n) log-probabilities; logsumexp over valid runners is 0.
    """
    fam = _as_base(base)
    if not (isinstance(points, int) and points >= 3):
        raise ValueError(f"points must be an int >= 3; got {points!r}")
    if not (0.0 < float(delta) < 0.5):
        raise ValueError(f"delta must be in (0, 0.5); got {delta!r}")
    mu_r, sd_r, mask_r, lead, n = _prepare(mu, sd, mask)
    if mask is None:
        mask_r = None
    if lattice is None:
        x = _lattice_rows(mu_r.detach(), sd_r.detach(), mask_r, fam, points,
                          delta)
    else:
        lattice = torch.as_tensor(lattice, dtype=mu.dtype, device=mu.device)
        x = lattice.expand(*lead, lattice.shape[-1]).reshape(
            -1, lattice.shape[-1]).detach()
    if backward == "analytic":
        out = _RaceLogProb.apply(mu_r, sd_r, x, mask_r, fam, bool(checkpoint))
    elif backward == "autograd":
        out = _forward_rows(mu_r, sd_r, mask_r, x, fam)[0]
    else:
        raise ValueError(
            f'backward must be "analytic" or "autograd"; got {backward!r}')
    return out.reshape(*lead, n)
