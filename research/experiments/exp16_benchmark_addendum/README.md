# Experiment 16: benchmark addendum

The outputs the paper references beyond exp13, run for the second referee
round.

**Parts A+B — metrics and the direct-MC baseline** (`run_addendum.py`, exp13's
problem sequence, fresh 2e6-draw truths):

| N | lattice max | lattice mean | lattice TV | lattice s | direct-MC max at matched time |
|---|---|---|---|---|---|
| 5 | 4.2e-4 | 1.8e-4 | 4.4e-4 | 0.02 | 5.4e-4 (0.3M draws) |
| 20 | 7.6e-4 | 1.2e-4 | 1.2e-3 | 0.09 | 1.2e-3 (0.6M) |
| 50 | 4.1e-4 | 5.6e-5 | 1.4e-3 | 0.23 | 5.5e-4 (0.6M) |
| 200 | 3.7e-4 | 2.7e-5 | 2.7e-3 | 0.86 | 7.0e-4 (0.6M) |
| 1000 | 4.9e-4 | 6.3e-6 | 3.1e-3 | 4.40 | 7.2e-4 (0.6M) |

**The honest headline**: direct utility simulation (argmax over draws — whole
share vector per draw, no per-alternative cost) at wall time matched to the
lattice call lands within a factor of 1.3–3 of the lattice's max-coordinate
error at every tested N. At the 1e-3 accuracy level, raw forward speed does
not separate the methods; error decay (R^{-1/2} vs lattice refinement),
reproducibility, resampling-free derivatives, and inversion do. This is in
the paper's abstract, as the referee required.

**Part C — inversion replication** (N=1000, k=2, three independent 5e6-draw
targets): 58–61 s each, the same 162 alternatives identified each time (98.1%
of share mass), recovery max 0.018/0.025/0.018, median ≈0.0026. Figure:
`figures/recovery_vs_share.png`.

**Part D — replication over problems** (`run_replication.py`, common spread
1.0, 10 problems per N at 20/50/200 and 3 at 1000, twin independent 2e6-draw
references per problem): worst-case lattice max error stayed below the
references' own replicate-to-replicate noise at every size (see
`results_replication.csv`).

Outputs: `results.csv`, `results_replication.csv`,
`figures/recovery_vs_share.png`.
