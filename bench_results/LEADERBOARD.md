# Arena leaderboard

Best wall time to reach each accuracy band (max-coordinate error vs the cached reference; the reference carries its own ~7e-4 noise, so sub-noise entries are reference-limited).

## n1000k2 (N=1000, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.237 | sobol_direct | 1.709 |
| 5e-4 | **lattice** | 0.237 | sobol_direct | 1.709 |

## n200k2 (N=200, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.047 | sobol_direct | 0.336 |
| 5e-4 | (none qualified) | | | |

## n200k3 (N=200, k=3)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **sobol_direct** | 0.334 | lattice | 0.442 |
| 5e-4 | **lattice** | 0.442 | direct_mc | 1.580 |

## n50k2 (N=50, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.012 | direct_mc | 0.037 |
| 5e-4 | (none qualified) | | | |

