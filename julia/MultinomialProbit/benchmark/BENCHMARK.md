# Fishing, cross-language (mlogit's vignette: J = 4, T = 1182)

Same model everywhere: mode ~ price + catch with alternative
intercepts, rank-2 zero-reference loadings, unit idiosyncratic
variance. Every fitted parameter vector is scored by ONE shared
referee — the python stabilized Sobol evaluation (two scrambles at
2^15) — because the recorded honesty note stands: every optimizer's
own likelihood carries 1-2 nats of integration or simulation noise at
the fitted sharpness, so per-engine numbers do not order optima.

Reproduce:
    Rscript -e 'data(Fishing, package="mlogit"); write.csv(Fishing, "fishing.csv", row.names=FALSE)'
    python julia/MultinomialProbit/benchmark/fishing_python.py fishing.csv out.json
    julia  julia/MultinomialProbit/benchmark/fishing_julia.jl out.json jl.json

Measured (2026-09-06, Apple Silicon, single process):

| engine | referee logLik | wall | converged |
|---|---|---|---|
| R mlogit GHK (recorded)      | -1215.7 (own report) | 22.2 s | — |
| R mlogit_fast (recorded)     | -1214.6 (own report) | 17.2 s | — |
| python MNProbit exact        | **-1212.82 ± 0.10** | 21.8 s | yes |
| julia MNProbit `:exact`      | -1214.82 ± 0.06 | **13.5 s** | yes |
| julia MNProbit `:ghk` (200 draws, CRN) | -1214.89 ± 2.26 | 65.3 s | no (60-iter cap) |

Readings, in honesty order:

1. The julia exact engine lands in the R-exact band, is the fastest
   wall clock in the table, and beats its own in-package GHK on every
   axis at once: ~5x the speed, a 40x tighter referee band, and
   convergence. (First-published julia timings were 87.9 s / 396.1 s;
   profiling attributed the gap to the dependency-free series normal
   CDF at 78 ns/call vs scipy's compiled Cephes at 10 ns -- the
   per-evaluation ratio equaled the primitive's ratio, acquitting the
   rest of the port. The series was replaced by a Julia port of the
   Cephes rationals themselves, oracle-tested to 4e-15, in all three
   Julia packages.)
2. The 2-nat gap to python's optimum is real and unexplained only in
   part: past sharpness 3 the two implementations escalate to
   DIFFERENT node families (python scrambled Sobol, julia
   dependency-free Halton, both 2^10), so above that threshold they
   optimize slightly different objectives; julia is stationary on its
   own. The recorded boundary-adjacency of Fishing's unrestricted
   likelihood means small node differences move the endpoint nats.
   python remains the reference implementation, as its own docs claim.
3. GHK's simulated -1210.93 at the fit illustrates the classic
   optimism of simulated likelihood (the referee places it at
   -1214.89 ± 2.26): 200 CRN draws flatter the objective by ~4 nats.
