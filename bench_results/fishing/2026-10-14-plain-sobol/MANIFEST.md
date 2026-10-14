# Run manifest

- suite dir: bench_results/fishing/2026-10-14-plain-sobol (fix3-sobol-sym: plain Sobol in the MNP sharp branch, owner decision on PR #719)
- main = origin/main 988e52cd (worktree ../winning-fix3-sobol-sym-main); branch = the same SHA plus the uncommitted sharp-cloud change, committed together with this directory (the "branch" SHA printed in stdout.txt is therefore the parent).
- command: `run.sh <main tree> main; run.sh <branch tree> branch` > stdout.txt. run.sh runs the protocol of ../2026-10-13-benchfix/MANIFEST.md (fishing_python.py, fishing_julia.jl, fishing_referee.py) and then score.py (sharpness and the Gauss-Hermite 151^3 truth at each exact fit).
- times.txt: `time.sh <main tree> <branch tree>`, three alternating rounds of the python exact fit and a warmed julia exact fit (ghk skipped).
- cloud_error.txt: `halton_error.py py_main.json jl_branch.json py_branch.json`, quadrature value of the sharp-branch clouds at three fitted points against the GH 151^3 truth.
- start 2026-10-14 08:02:50 EDT load 14.79; end of stdout.txt 08:13:27 EDT load 11.69; times.txt and cloud_error.txt after.
- scheduling: nice -n 10; OMP/OPENBLAS/MKL/VECLIB/RAYON/JULIA threads = 4
- machine: Apple M3 Ultra, 28 cores, 96 GB; Darwin 25.2.0 arm64
- python: Python 3.14.6 (/Users/petercotton/github/winning-bench-oct13/.venv); julia 1.12.6
- data: fishing.csv, sha256 in fishing_csv.sha256 (same file as ../2026-10-13)

## Results (main -> branch)

| fit | referee logLik | GH 151^3 truth | sharpness | fit time (median of 3) |
|---|---|---|---|---|
| python exact | -1216.46 -> **-1212.82** | -1216.30 -> -1212.87 | 2.88 -> 10.25 | 14.6 s -> 9.6 s |
| julia exact | -1216.26 -> -1216.24 | -1216.30 -> -1216.28 | 2.88 -> 2.89 | 15.5 s -> 12.1 s |

Julia does not reach the sharp optimum: its sharp cloud is a plain
(unscrambled) Halton set, which is 6.3 nats off at python's optimum
(-1219.12 against a truth of -1212.87, cloud_error.txt), so julia's
objective never sees the sharp basin as better. For comparison the R
ports' scrambled Sobol (r/*/R/sobol.R, .sobol_normal(3, 1024, seed 0),
evaluated with python's likelihood) gives -1219.73 / -1219.31 / -1208.16
at the three points (truth -1216.30 / -1216.28 / -1212.87). Before #605 julia landed at
-1214.52 (../2026-10-13-bisect/BISECT.md), never at -1212.8.
