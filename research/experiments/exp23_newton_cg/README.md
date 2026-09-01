# Experiment 23: Newton-Krylov calibration — three rounds, measured

Can the exact matrix-free Jacobian-vector product beat the production
damped-Jacobi (own-slope) iteration? Three formulations, in order of
increasing fairness to Newton.

## Round 1 (`run_newton_cg.py`): hobbled, loses badly

Plain UNpreconditioned CG (an earlier docstring claimed Jacobi
preconditioning that the loop never applied) on RAW share residuals with
the grid-form JVP.

| n=200 | time | iters | residual | converged |
|---|---|---|---|---|
| damped Jacobi (production) | 3.7s | 12 | 4.8e-7 | yes |
| Newton-CG round 1 | 387s | 25 Newton / 200 JVP | 2.1 | no |

Raw-share residuals make tiny-share coordinates invisible (both their
residuals and their Jacobian entries are tiny), so CG pours the step
into weakly determined tail directions.

## Round 2 (`run_newton_cg2.py`): partial fixes, still loses

Log residual driving the right-hand side and a diagonal preconditioner,
but still the grid-form (not-exactly-symmetric) JVP, no null-vector
projection, no symmetrization. Diverges: 418s, residual 34 at n=200.

## Round 3 (`run_newton_cg3.py`): the full recipe, first convergence

Log-residual Newton system L delta = P g~, symmetrized as
(P^-1/2 L P^-1/2) y = P^1/2 g~; IBP-form JVP (the explicitly symmetric
Laplacian operator); PCG with the own-log-slope diagonal -J_ii/p_i;
sqrt(p) null vector projected from rhs and iterates; trust clip; accept
only on true forward-residual improvement with Jacobi-sweep fallback;
four production sweeps as warm start.

| n=200 | jacobi | hybrid |
|---|---|---|
| easy (exp23 shape) | 0.5s, res 9.2e-9, converged | 29.7s, res 9.6e-10, converged |
| hard (strong correlation) | 1.3s, res 1.1e-4, STALLED at cap | 73s, res 6e+2, diverged |

## Verdict

The formulation fixes are real: round 3 is the first Krylov variant to
converge at all, and on the easy problem it reaches an order of
magnitude tighter residual than Jacobi -- at sixty times the wall clock,
because each IBP JVP costs about one forward pass and the easy problem
only needed twelve of those in total. On the strongly correlated
problem, where a coordinated step should pay, BOTH methods currently
fail: Jacobi stalls at 1.1e-4 against its iteration cap, and the hybrid
runs away despite the accept-test (the Jacobi fallback itself oscillates
there). So the promising target identified by the analysis -- correlated
inversions where Jacobi zig-zags -- is exactly where the remaining
engineering lives: Levenberg regularization, tail/active-set handling,
and a fallback that is itself stable on the hard set. Tracked in the
repository issues.

The SPD structure is verified elsewhere (exp17 Part C); nothing here is
a structure failure. The production calibration results in the paper use
the Jacobi iteration throughout, and the paper says so.

Run: `python run_newton_cg.py` / `run_newton_cg2.py` / `run_newton_cg3.py`.
