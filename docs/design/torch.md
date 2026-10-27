# winning.torch — Thurstone heads for PyTorch

Status: first cut (independent race with analytic backward, rank 1–2
factor race, three losses). Optional: `pip install 'winning[torch]'`.
`import winning` never imports torch.

## Purpose

Let a network emit **locations** (and optionally factor loadings) and
train on a **Thurstonian likelihood** in place of softmax / Plackett–Luce
/ Bradley–Terry. Two uses in view: a Thurstone output layer over a
candidate shortlist, and a Thurstonian reward model replacing
Bradley–Terry in preference tuning. Evaluate them **under restriction**
(second choices, larger menus), not by held-out loss alone.

## API

All min-wins, as in the rest of winning (lower `mu` wins; a head that
emits utilities passes `mu = -u`). All take leading batch dims.

| function | returns | backward |
|---|---|---|
| `race_log_prob(mu, sd=None, *, base="normal"\|"gumbel", points=257, delta=1e-12, mask=None, lattice=None, backward="analytic", checkpoint=False)` | `(..., n)` log P(i wins) | **analytic** O(n·L) VJP |
| `race_log_prob_factor(mu, V, sd=None, *, nodes=15, ...)` | `(..., n)`, loadings `V (..., n, rank)` | analytic per node; autograd for the node mixture |
| `thurstone_nll(mu, winner, sd=None, *, V=None, reduction="mean")` | −log P(winner) | analytic |
| `ordered_prefix_log_prob(mu, order (..., k), sd=None)` | log P(order[0] 1st, …, order[k-1] k-th) | autograd |
| `topk_log_prob(mu, k, sd=None)` | `(..., n)` log P(i among the k best) | autograd |
| `race_lattice(mu, sd, ...)` | `(..., points)` the win lattice (prefix / top-k use the wider event lattice below) | — |

* **Shapes.** `mu (..., n)`; `sd` `None`, scalar or broadcastable to
  `mu`; gradients flow to `mu` and `sd` (parameterise `sd = exp(s)`
  yourself). `base="gumbel"` is the standardized min-Gumbel: the race is
  then exactly `log_softmax(-π mu / (√6 sd))` — a built-in control (and
  `ordered_prefix_log_prob` under it is exactly Plackett–Luce).
* **Variable n.** Pad to `n_max` and pass `mask (..., n)` (False =
  padding). Padding runners do not run: −inf log-prob, zero gradient,
  and the valid runners' values equal the unpadded race's (tested).
* **Factor race.** `X = mu + V f + sd Z`, `f ~ N(0, I_rank)`, a fixed
  probabilists' Gauss–Hermite tensor rule (`nodes**rank` conditional
  races). The node axis is a batch axis for `race_log_prob`; autograd
  only sees `mu + V f_q` and the final `logsumexp` over nodes. `V` must
  be `(..., n, rank)` exactly (with batch dims `(rank, n)` is
  ambiguous), so it is exempt from the shape-spelling sweep.

## The forward

Per row, on a uniform lattice `x_1..x_L`:

    g_i(x) = log f_i(x) + Σ_{j≠i} log S_j(x)          (log win integrand)
    A_i    = logsumexp_x g_i(x)
    log p_i = A_i − logsumexp_j A_j

The rectangle weight `dx` is common and cancels in the normalisation —
the same rule as `winning.factor.race_probabilities`, which also divides
by the lattice total. Everything is in the log domain: `log S` is
`log_ndtr(−z)` (normal) or `−e^u` (Gumbel), so there is no `1 − Φ`
cancellation and no underflow of a longshot's share. The leave-one-out
`Σ_{j≠i} log S_j` is a subtraction, `LS − log S_i`; in the log domain this
is accurate to `eps·|LS|` and only where runner i's own integrand is
negligible does `log S_i` dominate `LS`, so prefix/suffix products (needed
in the *linear* domain to avoid dividing by an underflowed `S_i`) are not
needed here.

## The backward (why it is cheap)

With upstream gradient `G`:

    c_i  = G_i − p_i Σ_j G_j
    r_i(x) = c_i · softmax_x(g_i)(x)                  (n × L)
    ∂L/∂log f_k(x) = r_k(x)
    ∂L/∂log S_k(x) = Σ_{i≠k} r_i(x) = R(x) − r_k(x)
    ∂L/∂z_k(x) = r_k · (log f)'(z) + (R − r_k) · (log S)'(z)
    ∂L/∂mu_k = −Σ_x ∂L/∂z_k / sd_k,
    ∂L/∂sd_k = −(Σ_x z_k ∂L/∂z_k + c_k) / sd_k

`(log S)'` is minus the hazard, `exp(log f − log S)`, from the logs it
already holds. One O(n·L) pass; the n(n−1) cross terms of the Jacobian
are never formed (the column sum `R(x)` carries them). The function saves
`log S` (one `(R, n, L)` tensor) so the backward never re-evaluates
`log_ndtr`; `checkpoint=True` saves nothing of that size and recomputes.

**First-order only.** `A` and `log S` are computed in the forward without
a graph, so differentiating this backward again silently gave a wrong,
asymmetric Hessian (and `checkpoint=True` the same). The backward is
therefore first-order only: run with `create_graph=True` (a Hessian, a
gradient penalty) it raises a `RuntimeError` (#864). (`@once_differentiable`
was not enough: it only guards when the upstream gradient itself requires
grad, so `torch.autograd.functional.hessian` got a silent zero Hessian.) This covers everything that
calls it with the default `backward="analytic"`: `race_log_prob`,
`race_log_prob_factor` and `thurstone_nll`. `backward="autograd"`
(on `race_log_prob` and `race_log_prob_factor`) differentiates the
forward itself and passes `gradgradcheck`; for a second-order NLL gather
from that. The prefix and top-k losses are autograd throughout.

The lattice depends on `(mu, sd)` but is treated as a constant (computed
under `no_grad`). The exact integral does not depend on the lattice, so
the omitted term is the derivative of the discretisation error; gradcheck
passes with both a fixed and a moving lattice.

## The lattice (deterministic, vectorised, no search)

winning's `_bulk_window` bisects the winner distribution's quantiles
(~160 sequential scalar steps) and post-checks rare winners. On a GPU that
is the wrong shape. Here, per row, in closed form (float64, on the CPU for
MPS):

* **left:** `lo = min_i mu_i + sd_i q_lo`, `F(q_lo) = delta/n`; then
  **relative to each runner's own share**: 17 probes over the provisional
  window give the lower bound `p_i ≥ max_t F_i(t) Π_{j≠i} S_j(t)`, and `lo`
  moves left until `F_i(lo) ≤ delta·bound_i`. (Without this a noisy
  longshot behind a sharp favourite lost ~1e-8 of its share.)
* **right:** `hi = max_i min(mu_i + q_hi sd_i, m_i + 7.03 s_i)` where
  `(m_i, s_i)` is the Gaussian product of runner i with the lowest-mu
  other runner k. Rationale: runner i's integrand is `f_i S_k` times a
  non-increasing function, so its normalised right tail is dominated by
  that of `f_i S_k` (monotone likelihood ratio).
* `points` fixed (static shapes); `x = linspace(lo, hi, points)`.
* **scale-free:** the only width guard (for a one-runner row) is
  `hi ≥ lo + max_i sd_i`, relative to the row's own scale. An absolute
  `1e-12` floor there repriced `[0, 1]`, sd 1, from `[0.760, 0.240]` to
  `[0.681, 0.319]` when written in units of 1e-15 (#865); race, prefix
  and top-k are now invariant to 1e-12 under rescaling by 1e-15…1e15.

Cost O(17·n) per row. Accuracy measured against
`race_probabilities` (tests/test_torch_heads.py, float64, 257 points):

| field | max relative error of p_i |
|---|---|
| ordinary fields n = 3…100, both bases, hetero sd | ≤ 1e-11 |
| longshots to p = 1e-176 (`[0, 20, 40]`) | ≤ 1e-12 |
| `n = 1000`, `mu ~ N(0, 3²)`, sd ∈ [0.3, 3] | 1e-4 at 257 pts, 1e-13 at 513 |
| `mu` spread over 200 sd (n = 50) | 1e-9 at 257, 1e-13 at 513 |

The binding constraint is the **spacing**: errors are spectral once
`dx ≲ 0.3·min sd` and degrade past `0.5·min sd` (winning warns there; this
module does not — see open questions). A field spread over many sd, or a
near-deterministic runner, needs more points. On one gumbel field
(`[0, 20, 40]`, sd 0.6–1.8) this window is **more** accurate than
winning's default bulk window, whose rare-winner post-check is
normal-only (8e-6 relative off the converged span lattice; this module
3e-10, the span reference's own floor).

### Prefix and top-k: the event lattice (#753)

The win window is right for the win event only: it stops where every
**winner** integrand is negligible. A later place, or a top-k
membership, lives near its own runner, which can be far to the right
(on the win lattice P(0 then 1) at `mu = [0, 10]` came out 0.503, and
k = 3 orders of *ordinary* three-runner fields with sd 0.3–2 were off by
up to 88%). `ordered_prefix_log_prob` and `topk_log_prob` therefore use
an **event lattice** (`losses._event_lattice_rows`):

* **left:** the win lattice's (it already covers `min_i mu_i + sd_i q_lo`,
  relative to each runner's share).
* **right:** `hi = max(hi_win, max_i mu_i + q_hi · max_i sd_i)`,
  `S(q_hi) = delta`. The widest sd, not each runner's own: the last runner
  of a prefix must finish behind the others, so its mass sits at or left
  of about `max_i mu_i`, often deep in its own right tail (P(1 then 0) at
  `mu = [0, 5]`, sd `[1, 0.3]` is 8e-7, all near x = 4.6). For a
  log-concave base `S(z+q)/S(z) ≤ S(q)/S(0)` for `z ≥ 0`, so what is cut
  right of `hi` is ~delta *relative to the event*. (`mu_i + q_hi sd_i`
  per runner left 7e-7 relative error on that event.)
* **points:** `max(points, ⌈span / dx⌉ + 1)` with `dx = 0.3 · min sd`,
  capped at 4097 (or `points=`, if larger). For k ≥ 3 prefixes, whose
  nested cumulative integrals are fourth order rather than spectral, `dx`
  is also no coarser than the win lattice's at `points=`, up to `2·points`,
  so on ordinary fields they do not lose resolution to the wider window
  (~1.3–1.6× the points there).
  The count is the batch maximum (all rows share it) and reading it is
  one host sync per call. For k ≤ 2 and top-k, ordinary fields keep
  `points=`; separated or very heterogeneous fields get more.
* **past the cap** (span > ~1230 min sd) the window still covers every
  event and only the spacing grows: pairs and top-k sums stay at ~1e-11
  up to dx = 0.5 min sd (span ~2000 sd), ~3e-9 at dx = 1 sd, and 2e-2 at
  dx = 2 sd (span 8000 sd). Pass `points=` to go further.
* the prefix integrand's "rest" (`Σ log S` over runners not in the
  prefix) is summed directly: as `Σ_all − Σ_prefix` it cancelled where a
  prefix runner's `log S` is huge, which the wider window reaches (Gumbel
  `log S = −e^u` is ~−1e17 forty sd out).

`race_log_prob`, `thurstone_nll` and the factor race keep the win
lattice, bitwise unchanged.

**float32:** finite log-probs and gradients for 10- and 40-sd longshots
(both bases); absolute error of `p` ~1e-7 against float64. Use float64
for evaluation and anything that reads small probabilities.

## Cost

Memory per call ≈ `B·n·L` elements per live buffer, `× Q` for the factor
race (`Q = nodes**rank`). Time is dominated by `log_ndtr` on `B·n·L`
elements. CPU (Apple M-series, 4 threads) and MPS, points = 257, forward
+ backward, from `research/torch_heads/timing.py` (ms; MB = autograd's
saved tensors):

| device | dtype | B × n | analytic ms / MB | checkpoint ms / MB | autograd ms / MB |
|---|---|---|---|---|---|
| cpu | f32 | 256 × 10 | 5.0 / 2.8 | 6.7 / 0.3 | 5.3 / 15.1 |
| cpu | f32 | 64 × 100 | 10.6 / 6.4 | 15.1 / 0.2 | 11.5 / 37.7 |
| cpu | f32 | 8 × 1000 | 13.2 / 8.0 | 18.8 / 0.1 | 14.2 / 47.2 |
| cpu | f64 | 256 × 10 | 7.2 / 5.6 | 9.2 / 0.6 | 8.0 / 30.2 |
| cpu | f64 | 64 × 100 | 16.8 / 12.9 | 21.6 / 0.3 | 17.8 / 75.5 |
| cpu | f64 | 8 × 1000 | 20.9 / 15.9 | 27.0 / 0.3 | 22.2 / 94.4 |
| mps | f32 | 256 × 10 | 3.8 / 2.8 | 3.9 / 0.3 | 4.3 / 41.5 |
| mps | f32 | 64 × 100 | 4.2 / 6.4 | 4.7 / 0.2 | 5.2 / 103.6 |
| mps | f32 | 8 × 1000 | 4.8 / 8.0 | 5.2 / 0.1 | 5.6 / 129.5 |

Honest reading: the analytic backward wins **memory** (5–16× less saved
state; 100–1000× with `checkpoint=True`), and only ~6–23% of **time** —
autograd through the forward is also O(n·L), and `log_ndtr` in the forward
dominates both. The memory is what matters at scale (scaling the 64 × 100 row
to a shortlist head at B = 4096, n = 100: ~2.4 GB of tape under autograd
at f32, ~410 MB here, ~13 MB with checkpointing). MPS has no `log_ndtr`/`ndtri`/float64 in
torch 2.14: there `log_ndtr` is rebuilt from `erfcx`/`erfc` (float32
accuracy, same as CPU float32) and the window is computed on the CPU.

## Exact vs approximate

* Win probabilities, independent race: spectrally accurate lattice
  quadrature (as winning), ~1e-12–1e-11 relative at 257 points on ordinary
  fields; the gradient is the exact gradient of that forward.
* Factor race: fixed Gauss–Hermite order (default 15 per dimension), no
  adaptive node escalation; rank 1 matches winning to 1e-12 where winning
  also picks 15 nodes; sharp fields (loadings ≫ sd) need more nodes, and
  rank ≥ 3 is a `nodes**rank` blow-up (use winning's Sobol rule
  offline).
* `ordered_prefix_log_prob` (event lattice): k ≤ 2 spectral (`H_1 = F`
  exactly): ≤ 1e-11 relative against exact references on fields
  separated by 5–40 sd with sd 0.3–2, down to 1e-300 events (Gumbel:
  ~1e-9 on the rarest, its right flank is sharper than its sd). k = 3
  adds one cumulative trapezoid with an Euler–Maclaurin end correction,
  now across the whole wider span; the correction holds there (measured
  fourth-order convergence, 4× points → ~256× smaller error), but its
  constant grows with the log-slope of the cumulative integral's left
  tail, so the error is set by how rare the event is. On random
  three-runner fields with sd 0.3–2 at 257 points: ≤ 2e-7 relative for
  events above 5e-5 (1e-9 at 1025), up to ~1e-4 near 1e-12; 1e-3–5e-2
  for the 1e-45–1e-176 orders of a 10–20 sd field (7e-8–7e-6 at 4097). In log terms that is an absolute log-loss error
  of the same size. (winning's own `ordered_probabilities(k=3)` shows the
  same rare-event pattern: 5e-5 near 1e-12, 2e-3 below, 0.17 below
  1e-43.) Each extra
  level adds one such cumulative integral, so the **full permutation
  likelihood** (k = n) is available only through this nested recursion
  and its error grows with k — use short prefixes.
* `topk_log_prob` (event lattice): prefix/suffix Poisson-binomial count
  programs joined per runner (no deconvolution), ≤ 1e-10 relative vs
  exact / winning on ordinary to 40-sd fields, Σ = k to ~1e-12;
  O(n·L·k²), a Python loop over runners, autograd tape O(n·L·k). The
  count programs and their prefix/suffix join run in the **log domain**
  (`logaddexp` per count; log 0 is a finite stand-in so the backward has
  no NaN). The linear programs, clamped at `finfo.tiny`, floored a rare
  membership at log(tiny) — top-1 of a 20-sd longshot was −87.3 for
  −170.1 in float32, −708 for −1454 at a 60-sd gap in float64 — with
  near-zero gradients (#863). Now k = 1 equals `race_log_prob` in both
  dtypes and memberships far below 1e-308 keep their digits; on ordinary
  fields the values agree with the linear programs to rounding (~1e-15).

## Tests (tests/test_torch_heads.py; `importorskip("torch")`)

Parity with `race_probabilities` (both bases, hetero sd, longshots to
1e-176; tolerance 1e-10 relative because the windows differ), pair closed
form far into the tail, Gumbel = softmax, Σp = 1, `gradcheck` of the
analytic backward (fixed and moving lattice, masked, checkpointed),
analytic = autograd to 1e-12, scale-equivariance of the sd gradient,
padding invariance (values and gradients, both backward modes), variable
n batch = row-by-row, permutation equivariance, float32 tails
(finite values and gradients at 10 and 40 sd), factor race vs winning and
its gradcheck, zero loadings = independent race, ordered prefix vs
`ordered_probabilities` (k = 1, 2, 3) and = Plackett–Luce under Gumbel,
top-k vs `top_k_probabilities`, MPS float32 vs CPU float64 (if present),
and `import winning` not importing torch (subprocess). #753 (event
lattice): pair orders = closed form at gaps 5–40 sd and mixed sd; k = 2
and k = 3 orders of three runners and top-2 of three against exact
quadrature references; full orderings sum to 1 (n = 3, 4); top-k sums to
k (both bases); random fields (spread 1–25 sd) vs `ordered_probabilities`
/ `top_k_probabilities`; Plackett–Luce under Gumbel at 40 sd; mixed batch
= row by row; padding invariance; lattice size rule and cap; gradcheck,
float32 and MPS on separated fields.

## Limits and open questions

1. **CI has no torch job**, so these tests skip in CI. Proposed job (not
   made): ubuntu, py3.12, `pip install -e ".[test]" torch --index-url
   https://download.pytorch.org/whl/cpu`, run `pytest -q
   tests/test_torch_heads.py tests/test_port_surface.py
   tests/test_shape_contract.py`, and add `winning/torch` to the ruff step.
2. **Convention.** Min-wins everywhere, matching winning. A
   `higher_wins=True` switch on the losses would suit ML users; not added.
3. **Spacing warning.** winning warns when `dx > 0.5·min sd`; doing that
   here means a host sync per call. Option: a `race_lattice` diagnostic
   the caller checks once per epoch.
4. **Analytic backwards for prefix and top-k.** Both use autograd now; the
   same responsibility trick applies (prefix: the cumulative integrals are
   linear, so their adjoint is a reverse cumulative sum).
5. **Gauge.** `V` is not centred (winning gauge-fixes it); the race is
   invariant, but node placement is not. Centre in the head if it matters.
6. **Not here:** other bases (Student, skew), temperature/softmin, dense
   covariance, inversion (abilities from prices — use winning offline).
