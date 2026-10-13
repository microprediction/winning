# Arena leaderboard

Best wall time to reach each accuracy band (max-coordinate error vs the cached reference; the reference carries its own ~7e-4 noise, so sub-noise entries are reference-limited). Odds bands use max log-odds error over shares where the reference has ~2% relative noise or better (p >= 1.25e-3), the betting-relevant scale: 100-to-1 versus 150-to-1 is a huge difference carried by |dp| ~ 0.003.

## n1000k2 (N=1000, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.491 | sobol_direct | 1.537 |
| 5e-4 | **lattice** | 0.491 | sobol_direct | 1.537 |
| 10% odds | **lattice** | 0.491 | sobol_direct | 1.537 |
| 5% odds | **lattice** | 0.491 | - | nan |

## n200k2 (N=200, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.100 | sobol_direct | 0.306 |
| 5e-4 | (none qualified) | | | |
| 10% odds | **lattice** | 0.100 | direct_mc | 1.150 |
| 5% odds | **lattice** | 0.100 | genz_bretz | 9.159 |

## n200k3 (N=200, k=3)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **sobol_direct** | 0.333 | lattice | 0.899 |
| 5e-4 | **lattice** | 0.899 | direct_mc | 1.328 |
| 10% odds | **sobol_direct** | 0.333 | lattice | 0.899 |
| 5% odds | **lattice** | 0.899 | genz_bretz | 39.347 |

## n50k2 (N=50, k=2)

| band | winner | seconds | runner-up | seconds |
|---|---|---|---|---|
| 1e-3 | **lattice** | 0.025 | direct_mc | 0.040 |
| 5e-4 | (none qualified) | | | |
| 10% odds | **lattice** | 0.025 | mendell_elston | 0.031 |
| 5% odds | **lattice** | 0.025 | mendell_elston | 0.031 |

