# Experiment 23: Newton-CG calibration — negative result

Attempted: damped Newton-CG on the mean-zero quotient, matrix-free CG on the
SPD operator -J using the exact grid-form JVP, with the same
independent-inverse warm start as the production Jacobi solver.

Result (recorded honestly): it loses, badly.

| N=200 | time | iterations | residual | converged |
|---|---|---|---|---|
| damped Jacobi (production) | 11s | 12 | 4.8e-7 | yes |
| Newton-CG, cold start | 368s | 25 Newton / 184 JVPs | 2.6 | no |
| Newton-CG, warm start | 440s | 25 Newton / 200 JVPs | 2.1 | no |

Diagnosis: tiny-share coordinates have both tiny residuals and tiny
Jacobian entries, so CG pours the step into weakly-determined tail
directions; the step cap saturates and the iteration oscillates. The
production Jacobi solver survives exactly because of its tail-aware
tolerance and slope floors. A competitive Newton-Krylov needs a trust
region and restriction to the well-resolved block — future work, and the
paper claims only that the JVP *supports* Newton-Krylov methods, which
remains true; the reported calibration uses the Jacobi iteration.

The SPD structure itself is verified elsewhere (exp17 Part C: reduced -B'JB
positive definite), so this is a solver-engineering failure, not a
structure failure.

Run: `python run_newton_cg.py` (~15 min; expects the failure above).
