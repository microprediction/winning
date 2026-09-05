"""Laplacian structure of the ability-to-probability Jacobian.

For independent lattice performances X_i ~ f(. - a_i) with the *minimum*
winning (the package convention), the outright win probability of runner i is

    p_i(a) = sum_x f_i(x) * prod_{k != i} S_k(x),

where f_i is the shifted pmf and S_k(x) = Pr(X_k > x) the survival function.
Differentiating under the sum gives, for i != j,

    d p_i / d a_j = + w_ij,
    w_ij = sum_x f_i(x) f_j(x) prod_{k != i,j} S_k(x)  > 0,

and translation invariance p(a + c*1) = p(a) forces the diagonal
d p_i / d a_i = - sum_{j != i} w_ij.  Hence the Jacobian of the forward map
is minus a weighted complete-graph Laplacian:

    Dp(a) = -L(w),   L(w) = diag(W 1) - W.

(In the max-wins convention of random-utility theory the sign flips and
Dp = +L(w); only the orientation of "better" changes.)

The Laplacian need not be formed to be applied.  With the hazard
h_i(x) = f_i(x) / S_i(x) and the shared aggregates

    q_i(x) = f_i(x) * prod_{k != i} S_k(x),
    H(x)   = sum_j h_j(x),
    G_u(x) = sum_j h_j(x) u_j,

one has, for any vector u,

    (L u)_i = sum_{j != i} w_ij (u_i - u_j)
            = sum_x q_i(x) * (u_i H(x) - G_u(x)),

because the j = i term vanishes identically.  All aggregates are shared
across i, so a Hessian-vector product costs O(n M) for n runners on an
M-point lattice, versus O(n^2 M) to form the dense weights.  This enables
Newton-CG joint calibration without ever materialising the dense Jacobian.

Numerical strategy.  The hazard form fails where S_j(x) = 0 with
f_j(x) > 0 (atoms, edge pile-up, the top point of any truncated support)
and is ill-conditioned where S_j(x) is merely tiny: hazards up to f/S
enter the shared sums H and G_u, and the subtraction u_i H - G_u then
cancels catastrophically.  laplacian_matvec therefore splits the work:

- Hazards are used only where S_j >= SURVIVAL_TOL, bounding every hazard
  by 1/SURVIVAL_TOL and hence the cancellation error of a column by
  roughly n * machine_eps / SURVIVAL_TOL, i.e. ~1e-9 relative at the
  default tolerance.
- Every masked (j, x) with f_j(x) > MASS_TOL is then repaired *exactly*:
  the pair terms f_i f_j prod_{k != i,j} S_k it should have contributed
  to each row i are added via division-free prefix/suffix leave-one-out
  products, O(n) per masked point.  For smooth densities only a short
  band at the top of each runner's support is masked, so the total cost
  stays O(n M + n * #masked).
- Masked points with f_j <= MASS_TOL are dropped; the omitted mass is
  bounded by n * M * MASS_TOL * max|u_i - u_j| / unit, which is ~1e-11
  at the defaults.

The dense routine avoids division entirely and is exact by construction;
it is the reference against which the matvec is tested, including for
atoms, ties of atoms, zero-mass (off-lattice) runners, and adversarial
near-zero survival masses.

Degenerate fields disconnect the graph in two ways, both handled exactly:

- Zero-mass runners (sum p = 0, the package's off-lattice sentinel) have
  S = 1 and f = 0 everywhere: they leave the other weights untouched but
  contribute a zero row and column.
- Deterministically dominated runners (an atom that some other runner
  always beats) have their win probability frozen at a boundary face, so
  every weight involving them vanishes and they become isolated vertices.

Either way L acquires additional null vectors and lambda_2 = 0; callers
doing Newton steps should drop such runners first (their coordinates are
not identifiable from winner probabilities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .density import Density

# Below this survival the hazard f/S is not used; the point is repaired
# exactly instead. Keeping the threshold fairly large bounds hazards (and
# therefore floating-point cancellation) without any loss of accuracy,
# because the repair path is exact.
SURVIVAL_TOL = 1e-6

# Masked points with pmf mass at or below this are dropped outright; the
# resulting error is provably negligible (see module docstring).
MASS_TOL = 1e-15


def _validate_field(densities: Sequence[Density]) -> None:
    if len(densities) < 2:
        raise ValueError("Need at least two runners.")
    lattice = densities[0].lattice
    for d in densities[1:]:
        if d.lattice.L != lattice.L or d.lattice.unit != lattice.unit:
            raise ValueError("All densities must share the same lattice.")
    for d in densities:
        if not np.all(np.isfinite(d.p)) or np.any(d.p < 0.0):
            raise ValueError("Density pmf must be finite and non-negative.")


def _pmf_and_survival(densities: Sequence[Density]) -> tuple[np.ndarray, np.ndarray]:
    """Stack pmfs and survival functions, shapes (n, M)."""
    F = np.stack([d.p for d in densities])
    S = np.stack([1.0 - d.cdf() for d in densities])
    return F, np.clip(S, 0.0, 1.0)


def _leave_one_out_products(S: np.ndarray) -> np.ndarray:
    """loo[i] = prod_{k != i} S[k], columnwise, via prefix/suffix products.

    Division-free, so exact zeros in S are handled correctly. O(n M).
    """
    n, M = S.shape
    prefix = np.ones((n + 1, M))
    np.cumprod(S, axis=0, out=prefix[1:])
    suffix = np.ones((n + 1, M))
    np.cumprod(S[::-1], axis=0, out=suffix[1:])
    return prefix[:n] * suffix[:n][::-1]


def outright_win_probabilities(densities: Sequence[Density]) -> np.ndarray:
    """No-tie win probabilities p_i = sum_x f_i prod_{k != i} S_k, in O(n M).

    On a lattice ties carry positive mass, so the sum over i falls short of
    one by the total tie probability; this is the smooth forward map whose
    Jacobian is -L(w), not the dead-heat-adjusted state price of Race.
    """
    _validate_field(densities)
    F, S = _pmf_and_survival(densities)
    return np.sum(F * _leave_one_out_products(S), axis=1)


def laplacian_weights(densities: Sequence[Density]) -> np.ndarray:
    """Dense symmetric weight matrix w_ij = sum_x f_i f_j prod_{k != i,j} S_k.

    Division-free reference implementation: for each i, leave-one-out
    products are rebuilt over the remaining runners. O(n^2 M). Exact for
    atoms, zero-mass runners, and any other degenerate pmf.

    Lattice pmfs are masses (density times unit), so one factor of the
    lattice unit is divided out to make w_ij a derivative with respect to
    physical ability: dp_i/da_j = +w_ij for i != j.
    """
    _validate_field(densities)
    F, S = _pmf_and_survival(densities)
    n = F.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        rest = [k for k in range(n) if k != i]
        loo_rest = _leave_one_out_products(S[rest])  # (n-1, M): prod over rest minus one
        W[i, rest] = (F[rest] * loo_rest) @ F[i]
    W /= densities[0].lattice.unit
    return 0.5 * (W + W.T)  # symmetric up to roundoff; enforce exactly


def laplacian_dense(densities: Sequence[Density]) -> np.ndarray:
    """L(w) = diag(W 1) - W. The Jacobian of outright_win_probabilities is -L."""
    W = laplacian_weights(densities)
    return np.diag(W.sum(axis=1)) - W


class LaplacianOperator:
    """Applies L(w) for a fixed field, with the field structure precomputed.

    Construction costs O(n M + n * #masked) and captures everything that
    does not depend on u: the leave-one-out survival products q_i, the
    masked hazard aggregates, and the exact repair masses for points where
    the hazard identity fails (see laplacian_matvec).  Each subsequent
    matvec is then a handful of elementwise passes over the (n, M) arrays,
    which is what makes conjugate-gradient loops cheap: the field is
    analysed once per Newton step, not once per CG iteration.
    """

    def __init__(self, densities: Sequence[Density]):
        _validate_field(densities)
        F, S = _pmf_and_survival(densities)
        self.n = F.shape[0]
        self.unit = densities[0].lattice.unit
        self._Q = F * _leave_one_out_products(S)  # q_i(x), division-free
        masked = S < SURVIVAL_TOL
        self._h = np.where(masked, 0.0, F) / np.where(masked, 1.0, S)
        self._H = self._h.sum(axis=0)

        # Exact repair: a masked hazard h_j at column x removed the pair
        # term f_i f_j prod_{k != i,j} S_k from every row i != j.  Rebuild
        # those terms without division via leave-one-out products over
        # k != j, batched per runner; only the u-independent pair masses
        # need to be stored.
        self._repairs: list[tuple[int, np.ndarray, np.ndarray]] = []
        bad_j, bad_x = np.nonzero(masked & (F > MASS_TOL))
        keep = np.ones(self.n, dtype=bool)
        for j in np.unique(bad_j):
            cols = bad_x[bad_j == j]
            keep[j] = False
            loo2 = _leave_one_out_products(S[keep][:, cols])  # (n-1, |cols|)
            pair_mass = (F[keep][:, cols] * F[j, cols][None, :] * loo2).sum(axis=1)
            self._repairs.append((int(j), np.flatnonzero(keep), pair_mass))
            keep[j] = True

    def matvec(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.shape != (self.n,):
            raise ValueError("u must have one entry per runner.")
        if not np.all(np.isfinite(u)):
            raise ValueError("u must be finite.")
        G = (self._h * u[:, None]).sum(axis=0)
        out = np.sum(self._Q * (np.outer(u, self._H) - G[None, :]), axis=1)
        for j, idx, pair_mass in self._repairs:
            out[idx] += pair_mass * (u[idx] - u[j])
        return out / self.unit

    __call__ = matvec


def laplacian_matvec(densities: Sequence[Density], u: np.ndarray) -> np.ndarray:
    """Apply L(w) to u in O(n M + n * #masked) without forming the weights.

    Uses (L u)_i = sum_x q_i (u_i H - G_u) wherever survivals are healthy,
    plus exact division-free repairs where they are not; see the module
    docstring for the error analysis.  Agrees with laplacian_dense to
    floating-point accuracy for every field the package can represent,
    including atoms, edge pile-up, and zero-mass runners.

    The constant vector is annihilated exactly: G_u is accumulated in the
    same reduction order as H, so u = 1 yields identical floats and the
    integrand is exactly zero, as are the repair terms u_i - u_j.

    For repeated products with the same field (e.g. conjugate gradients),
    construct a LaplacianOperator once instead.
    """
    return LaplacianOperator(densities).matvec(u)


def _cg_mean_zero(matvec, b: np.ndarray, iters: int, tol: float = 1e-14) -> np.ndarray:
    """Conjugate gradients for L x = b, b mean-zero, L PSD with null vector 1.

    Every Krylov vector stays in the mean-zero subspace because L maps into
    it, so the translation gauge never contaminates the solve.  A
    non-positive curvature p'Lp (a disconnected graph direction) terminates
    the iteration with the best iterate so far.
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    for _ in range(iters):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0.0:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


@dataclass
class InversionResult:
    """Outcome of invert_outright_probabilities.

    abilities are mean-centered (the translation gauge is fixed at zero
    mean).  residual is the max-norm of achieved - scale * target, where
    scale = sum(achieved) / sum(target): only probability *ratios* are
    matched, with one multiplicative renormalization absorbing the total
    the no-tie lattice map can actually reach (one minus its tie mass).
    Matching ratios rather than differences keeps longshots meaningful --
    an additive offset criterion would declare a 1e-6 target "met" by a
    1e-3 probability.  The renormalization is exact for Newton purposes:
    the rescaled residual's Jacobian is exactly -L(w) because the
    Laplacian's columns sum to zero, and the residual itself sums to zero
    by construction.

    message states why the solver stopped; residual_history records the
    residual at the start and after each accepted Newton step.  When the
    message mentions the lattice edge, some ability sits at the largest
    shift the lattice can represent: the target needs a wider spread than
    L * unit allows, and the gauge may then deviate slightly from exact
    zero mean.
    """

    abilities: np.ndarray
    achieved: np.ndarray
    residual: float
    iterations: int
    converged: bool
    message: str = ""
    residual_history: np.ndarray | None = None
    scale: float = 1.0


def invert_outright_probabilities(
    bases: Sequence[Density],
    target: Sequence[float],
    *,
    initial: Sequence[float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    cg_iters: int = 200,
) -> InversionResult:
    """Joint Newton-CG inversion: win probabilities -> abilities, O(n M) per step.

    Finds mean-zero abilities a such that shifting each base density by a_i
    reproduces the target outright win probabilities up to the single
    multiplicative renormalization discussed in InversionResult (so targets
    may be given as normalized market prices).  Each Newton step solves
    L(w) delta = residual with conjugate gradients on a precomputed
    LaplacianOperator, so the dense Jacobian is never formed; a
    backtracking line search on the squared residual guards against
    overshoot (the analytic Laplacian is an O(unit) approximation of the
    discrete map's derivative).

    bases are the *unshifted* per-runner performance densities; they may be
    heterogeneous.  Requirements and failure modes:

    - Every base must carry mass, and the target must be strictly positive
      and finite: a zero-mass runner or a boundary target is not attainable
      at finite abilities (raises ValueError).
    - Degenerate fields whose graph disconnects mid-iteration
      (deterministic domination, lambda_2 = 0) stall the solver; the
      result is then returned with converged = False and a message saying
      why.
    - Abilities are clipped to the largest shift the lattice represents
      (shift_fractional saturates beyond +/- (L-2) steps, deadening that
      coordinate).  A solution pinned at the clip means the target needs
      more ability spread than the lattice affords; the message says so
      and the fix is a wider lattice (larger L * unit).
    - When the plain Newton direction fails its line search (the analytic
      Laplacian mismatches the discrete map's derivative, e.g. near
      atoms), ridge-regularized directions (L + lambda I) delta = r are
      tried at escalating lambda before giving up: these bend the step
      toward steepest descent of the squared residual, which the exact
      Jacobian range guarantees is a descent direction.
    - Atoms make the forward map only piecewise smooth; the solver may
      still converge but no guarantee is made.
    """
    _validate_field(bases)
    n = len(bases)
    for d in bases:
        if float(np.sum(d.p)) == 0.0:
            raise ValueError("Zero-mass base density: abilities are not identifiable.")
    target = np.asarray(target, dtype=float)
    if target.shape != (n,):
        raise ValueError("target must have one entry per runner.")
    if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
        raise ValueError("target probabilities must be finite and strictly positive.")

    lattice = bases[0].lattice
    unit = lattice.unit
    # largest shift shift_fractional can represent before saturating
    box = (lattice.L - 2) * unit
    if initial is None:
        a = np.zeros(n)
    else:
        a = np.asarray(initial, dtype=float).copy()
        if a.shape != (n,) or not np.all(np.isfinite(a)):
            raise ValueError("initial must be a finite vector with one entry per runner.")
    a = np.clip(a - a.mean(), -box, box)

    def build(ab: np.ndarray) -> list[Density]:
        return [d.shift_fractional(ai / unit) for d, ai in zip(bases, ab)]

    target_sum = float(target.sum())

    def scaled_residual(ab: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """achieved, achieved - scale * target, scale.

        The residual sums to zero by construction, and its Jacobian with
        respect to abilities is exactly -L(w): the renormalization term
        involves 1'L, which vanishes.
        """
        achieved = outright_win_probabilities(build(ab))
        scale = float(achieved.sum()) / target_sum
        return achieved, achieved - scale * target, scale

    achieved, r, scale = scaled_residual(a)
    res = float(np.abs(r).max())
    history = [res]
    iterations = 0
    message = ""

    def line_search(delta: np.ndarray) -> bool:
        """Accept the first damped step that reduces the L2 residual."""
        nonlocal a, achieved, r, res, scale
        res2 = float(r @ r)
        step = 1.0
        for _ in range(25):
            trial = np.clip(a + step * delta - np.mean(a + step * delta), -box, box)
            achieved_t, r_t, scale_t = scaled_residual(trial)
            if float(r_t @ r_t) < res2:
                a, achieved, r, scale = trial, achieved_t, r_t, scale_t
                res = float(np.abs(r).max())
                return True
            step *= 0.5
        return False

    while res >= tol and iterations < max_iter:
        iterations += 1
        op = LaplacianOperator(build(a))
        delta = _cg_mean_zero(op.matvec, r, cg_iters)
        if not np.all(np.isfinite(delta)) or not np.any(delta):
            message = (
                "Newton direction vanished: the interaction graph is disconnected "
                "(zero-mass, deterministically dominated, or off-lattice runner)."
            )
            break
        if not line_search(delta):
            # Ridge fallback: bend the direction toward steepest descent of
            # the squared residual, scaled by the curvature along r.
            curvature = float(r @ op.matvec(r)) / float(r @ r)
            accepted = False
            if curvature > 0.0:
                for lam in (1e-3 * curvature, 0.1 * curvature, 10.0 * curvature):
                    delta = _cg_mean_zero(lambda v: op.matvec(v) + lam * v, r, cg_iters)
                    if line_search(delta):
                        accepted = True
                        break
            if not accepted:
                message = "Line search stalled even with ridge regularization."
                break
        history.append(res)

    converged = res < tol
    if converged:
        message = "converged"
    elif not message:
        message = "max_iter reached before the tolerance was met."
    if float(np.abs(a).max()) >= box * (1.0 - 1e-12):
        message += (
            " Some ability sits at the lattice clamp: the target requires more "
            "ability spread than this lattice represents (enlarge L * unit)."
        )

    return InversionResult(
        abilities=a,
        achieved=achieved,
        residual=res,
        iterations=iterations,
        converged=converged,
        message=message,
        residual_history=np.asarray(history),
        scale=scale,
    )
