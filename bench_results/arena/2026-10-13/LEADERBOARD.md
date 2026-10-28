# Arena leaderboard

Best wall time to reach each accuracy band (max-coordinate error vs the cached reference; the reference carries its own ~7e-4 noise, so sub-noise entries are reference-limited). Odds bands use max log-odds error over shares where the reference has ~2% relative noise or better (p >= 1.25e-3), the betting-relevant scale: 100-to-1 versus 150-to-1 is a huge difference carried by |dp| ~ 0.003.

## n50k2 (N=50, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.026 | direct_mc | 0.040 |
| 5e-4 | (none qualified) | | | |
| 10% odds | **lattice** | 0.026 | sobol_direct | 0.085 |
| 5% odds | **lattice** | 0.026 | factor_rqmc | 0.102 |

