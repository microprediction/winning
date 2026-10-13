# mvtnorm (R) on the arena problems

Each share vector = N calls to pmvnorm(GenzBretz), the
standard MVN software, on the (N-1)-dim difference orthant.
Scored against the same cached references as the arena.

| problem | maxpts | seconds | max abs err | max log-odds err |
|---|---|---|---|---|
| n50k2 | 25000 | 3.4 | 7.3e-04 | 1.7e-02 |
| n50k2 | 250000 | 18.9 | 8.3e-04 | 1.8e-02 |
| n200k2 | 25000 | 112.3 | 7.1e-04 | 5.1e-02 |
| n200k2 | 250000 | 657.3 | 5.9e-04 | 4.6e-02 |
| n1000k2 | - | (skipped: 1000 calls of 999-dim pmvnorm is hours of wall time) | | |
| n200k3 | 25000 | 105.2 | 2.2e-04 | 4.3e-02 |
| n200k3 | 250000 | 645.5 | 2.7e-04 | 4.3e-02 |
