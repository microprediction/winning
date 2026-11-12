# Quiet-machine timings, 2026-11-11

Every wall-clock figure in the paper at tag paper-r12 comes from these logs.
Machine: Apple Mac Studio (M3 Ultra, 20 performance + 8 efficiency cores,
96 GB), macOS 26.2, the pinned environment (`requirements.lock`), one BLAS
thread. The lattice kernels use one core; the Rust GHK comparator
(fastrace) is parallel across alternatives and uses every free core.

- Window 1, 20:08-20:46 EST, load 0.16 per core: `retime.log`/`.json`
  (per_alternative, grammar_inversion, ghk, saturated, scale,
  invert_million, tenmillion), `quiet_alt_{1,2,3}.log` (`bench.py alt`,
  three runs; the paper quotes the median), `quiet_large_n.log`,
  `quiet_mle3.log` (covariates).
- Window 2, 21:21-21:41 EST, load 0.20 per core: `retime2.log`/`.json`
  (law, genz200, tilting, covariates repeat).
- `quiet_loop.log`, `quiet2.log`: the load at each attempt.

The genz200 R call was unseeded in window 2; it is seeded since
(`set.seed(20260827)`), which moves the Genz estimate within its error
estimate and not its time.
