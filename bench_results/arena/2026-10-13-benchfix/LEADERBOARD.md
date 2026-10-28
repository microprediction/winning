# Arena leaderboard

Best wall time to reach each accuracy band (max-coordinate error vs the cached reference; the reference carries its own ~7e-4 noise, so sub-noise entries are reference-limited). Odds bands use max log-odds error over shares where the reference has ~2% relative noise or better (p >= 1.25e-3), the betting-relevant scale: 100-to-1 versus 150-to-1 is a huge difference carried by |dp| ~ 0.003.

## n1000k2 (N=1000, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.492 | sobol_direct | 1.527 |
| 5e-4 | **lattice** | 0.492 | sobol_direct | 1.527 |
| 10% odds | **lattice** | 0.492 | sobol_direct | 1.527 |
| 5% odds | **lattice** | 0.492 | - | nan |

## n200k2 (N=200, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.099 | sobol_direct | 0.300 |
| 5e-4 | (none qualified) | | | |
| 10% odds | **lattice** | 0.099 | direct_mc | 1.110 |
| 5% odds | **lattice** | 0.099 | genz_bretz | 7.595 |

## n200k3 (N=200, k=3)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **sobol_direct** | 0.296 | lattice | 0.888 |
| 5e-4 | **lattice** | 0.888 | direct_mc | 1.162 |
| 10% odds | **sobol_direct** | 0.296 | lattice | 0.888 |
| 5% odds | **lattice** | 0.888 | genz_bretz | 37.599 |

## n50k2 (N=50, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.026 | direct_mc | 0.034 |
| 5e-4 | (none qualified) | | | |
| 10% odds | **mendell_elston** | 0.016 | lattice | 0.026 |
| 5% odds | **mendell_elston** | 0.016 | lattice | 0.026 |

