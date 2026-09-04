# Experiment 23: Newton-Krylov calibration — four rounds, measured

Can the exact matrix-free Jacobian-vector product beat the production
damped-Jacobi (own-slope) iteration? Three formulations, in order of
increasing fairness to Newton, then a fourth round checking whether the
n=200 verdict holds as field size grows.

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

## Round 4 (`run_field_size_scaling.py`): does it move with field size?

Round 3 was only ever run at n=200. A reader asked whether the wall-clock
gap would close as n grows toward the million-contestant regime the
production solver targets: production Jacobi's sweep count is set by a
spectral gap, and if that gap doesn't shrink with n while a JVP costs
about the same as a sweep, the verdict should flip well before a
million. Measured, n=200 to n=12,800:

| n | forward pass | JVP (ibp) | ratio | Jacobi sweeps |
|---|---|---|---|---|
| 200 | 128.7 ms | 1479.7 ms | 11.5x | 19 |
| 800 | 507.8 ms | 5885.7 ms | 11.6x | 18 |
| 3,200 | 2038.4 ms | 23684.3 ms | 11.6x | 23 |
| 12,800 | 8122.5 ms | 96357.1 ms | 11.9x | 23 |

An IBP-form JVP costs 11.5-11.9x a forward pass, not "about one" as the
verdict below used to say -- that was asserted, not measured. The ratio
is flat across a 64x range in n. So is Jacobi's own sweep count: this
correlation structure comes from a rank-2 factor model, so the
tie-density graph's effective diameter doesn't grow with n, and neither
does the sweep count needed to converge.

Run unmodified at n=3,200 -- same 10-Newton-step, 5-inner-CG budget that
converges cleanly at n=200 (8 Newton steps, 15 JVPs) -- the round 3
hybrid does not converge: 10 Newton steps, 20 JVPs, residual stuck at
2.4e-1 against 1e-8 tolerance. Wall clock: production Jacobi 46.8s, the
hybrid 515.6s and still short. The gap widens, not narrows. Whether
that's the outer Newton step count genuinely growing with n or the
trust-region/step-acceptance constants in `run_newton_cg3.py` being
tuned for n=200 and never rescaled isn't separated by this run -- either
way, the flat-inner-count mechanism that would flip the verdict doesn't
get a chance to run here, because the unmodified hybrid fails to
converge before field size is the limiting factor.

## Verdict

The formulation fixes are real: round 3 is the first Krylov variant to
converge at all, and on the easy problem it reaches an order of
magnitude tighter residual than Jacobi -- at sixty times the wall clock,
because each IBP JVP costs roughly twelve forward passes' worth of work
(round 4, measured) and the easy problem needed fifteen of those in
total. On the strongly correlated problem, where a coordinated step
should pay, BOTH methods currently fail: Jacobi stalls at 1.1e-4 against
its iteration cap, and the hybrid runs away despite the accept-test (the
Jacobi fallback itself oscillates there). So the promising target
identified by the analysis -- correlated inversions where Jacobi
zig-zags -- is exactly where the remaining engineering lives: Levenberg
regularization, tail/active-set handling, and a fallback that is itself
stable on the hard set, and now also making round 3 converge past
n=200. Tracked in the repository issues.

The SPD structure is verified elsewhere (exp17 Part C); nothing here is
a structure failure. The production calibration results in the paper use
the Jacobi iteration throughout, and the paper says so.

Run: `python run_newton_cg.py` / `run_newton_cg2.py` / `run_newton_cg3.py`
/ `run_field_size_scaling.py`.
