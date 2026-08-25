# Numerical test of the OT / Laguerre limit

Run 2026-08-19, `laguerre_limit.py`, rank r = 2, Sigma_tau = VV' + tau^2 I.

## What was tested

Two claims about the tau -> 0 limit of the factor-probit transform:

- **(A)** shares converge to Gaussian measures of power (Laguerre) cells,
  `p_i^tau -> gamma_2(C_i)`;
- **(B)** the share Jacobian converges to the weighted graph Laplacian of
  the power diagram, `dp_i/dmu_j -> -k_ij` with
  `k_ij = (1/|v_i - v_j|) * integral of phi_2 over the shared facet`
  -- i.e. the Newton Hessian of semi-discrete optimal transport.

In r = 2 the facet integral is closed form (see the module docstring), so
(B) can be checked against an exact reference instead of another solver.

## Results

**The closed-form edge weight is correct.** For N sites on a circle with
equal mu, the power diagram is a set of sectors, cell masses are exactly
1/N, and the Laplacian is exactly a cycle graph with edge weight
`phi(0)/2 / (2a sin(pi/N))`. The facet code reproduces that to **5.6e-17**.
On a random (non-symmetric) configuration it agrees with common-random-number
finite differences of a Monte Carlo mass estimate to **1.2e-03, i.e. 0.3% of
scale**. Two independent confirmations.

**(A) confirmed, rate about O(tau).** True probit shares (Monte Carlo, no
quadrature) against the exact Laguerre masses, random configuration:

| tau | max abs error |
|---|---|
| 0.500 | 5.3e-02 |
| 0.250 | 3.0e-02 |
| 0.125 | 1.3e-02 |
| 0.0625 | 5.1e-03 |

Halving tau roughly halves the error (ratios 1.8, 2.2, 2.6), so first order
in tau, no sign of a plateau.

**(B) supported, then defeated by quadrature.** Ring case, Jacobian error as
a fraction of scale, at the best quadrature order tried (Q = 81):

| tau | 0.500 | 0.250 | 0.125 | 0.0625 | 0.03125 |
|---|---|---|---|---|---|
| rel. error | 0.511 | 0.322 | 0.171 | 0.110 | 0.750 |

Monotone convergence toward the Laplacian at roughly O(tau^0.8) down to
tau ~ 0.06, then the error blows up: below that, product Gauss-Hermite over
F can no longer resolve a nearly discontinuous integrand. Raising Q delays
the wall but does not remove it (at tau = 0.0625 the error is 0.995 / 0.355 /
0.110 for Q = 15 / 41 / 81).

## The practical consequence

The bridge is **mathematical, not numerical**. The package's Gauss-Hermite +
lattice machinery is accurate where tau is moderate (at tau = 0.5, Q = 81 the
shares are exact to 5.7e-10) and degrades exactly where OT solvers are
strongest. So a benchmark against a semi-discrete OT solver should not chase
tau -> 0 agreement; the useful comparison is tau in roughly [0.05, 0.5] with
Richardson extrapolation in tau, or a change of F-quadrature (adaptive /
geometry-aware) if the hard limit is really wanted.

## Unplanned finding: empty cells and identifiability

Random sites with random weights generically produce **empty** Laguerre
cells: 50 attempts failed to find an 8-site configuration where every cell
had mass above 0.02. At tau = 0 a dominated alternative has probability
exactly zero, so its mu is unidentifiable (any sufficiently small value gives
the same shares) and the convex dual is not strictly convex there. Any
tau > 0 gives every alternative positive probability and restores
identifiability.

**Confirmed the hard way (2026-08-19).** The exact-reference OT solver
(`soft_laguerre.solve_ot_weights`, a damped Newton on the analytic Gaussian
mass map and its exact Laplacian) **failed to converge** on a random 8-site
configuration: 60 iterations, residual stuck at 5.9e-02, recovered mu off by
0.51. Cause: that configuration's target mass vector contained an exactly
empty cell (p_6 = 0) and a near-empty one (p_7 = 0.0017), so the Laplacian is
disconnected, the dual is not strictly convex, and mu_6 is unidentifiable.
This is not a solver bug to paper over -- it is the degeneracy, reproduced.

Practical consequence worth pursuing: the **soft problem can be used as a
continuation method for the hard one**. Solve the probit inverse at moderate
tau (well-conditioned, every alternative has positive probability), then
anneal tau downward using each solution as the warm start for the next. That
turns the identifiability observation into an algorithm, and it is the most
concrete practical claim the OT connection has produced so far. Untested.

This is the same condition the semi-discrete OT literature imposes for
Newton convergence (iterates must keep all cells non-empty). So the
idiosyncratic noise D is not merely a modelling nicety: it is a
**regularizer of a degenerate geometric inverse problem**, and that is a
point in favour of the smoothed object rather than against it.
