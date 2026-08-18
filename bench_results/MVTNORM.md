# mvtnorm (R) on the arena problems

Each share vector = N calls to pmvnorm(GenzBretz), the
standard MVN software, on the (N-1)-dim difference orthant.
Scored against the same cached references as the arena.

| problem | maxpts | seconds | max abs err | max log-odds err |
|---|---|---|---|---|
| n50k2 | 25000 | 3.1 | 8.4e-04 | 1.8e-02 |
| n50k2 | 250000 | 17.2 | 8.6e-04 | 1.8e-02 |
| n200k2 | 25000 | 85.8 | 6.1e-04 | 4.5e-02 |
| n200k2 | 250000 | 513.7 | 6.1e-04 | 4.7e-02 |
| n1000k2 | - | (skipped: 1000 calls of 999-dim pmvnorm is hours of wall time) | | |
| n200k3 | 25000 | 87.3 | 2.2e-04 | 4.6e-02 |
| n200k3 | 250000 | 719.5 | 2.6e-04 | 4.4e-02 |

Reading: mvtnorm is accurate -- its errors sit at the references' own
noise floor, same as the lattice, and do not improve with a tenfold
budget increase (both are reference-limited). The difference is wall
time: at N = 200 the lattice pass takes 0.05 seconds against 86-514
seconds here (a factor of 1800-10000), because each share vector costs N
separate (N-1)-dimensional integrals. Same answers, one pass versus N.
