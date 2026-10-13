# Run manifest

- suite dir: bench_results/fishing/2026-10-13-benchfix (fix3-benchfix: #716 #717 #718)
- command (repo root, PYTHONPATH=.): `python julia/MultinomialProbit/benchmark/fishing_python.py fishing.csv bench_results/fishing/2026-10-13-benchfix/py.json; julia julia/MultinomialProbit/benchmark/fishing_julia.jl bench_results/fishing/2026-10-13-benchfix/py.json bench_results/fishing/2026-10-13-benchfix/jl.json; python julia/MultinomialProbit/benchmark/fishing_referee.py bench_results/fishing/2026-10-13-benchfix/py.json bench_results/fishing/2026-10-13-benchfix/jl.json`
- git SHA (code under test): abf367b616f09cdf43f55fb05e0cc59f66961bcb
- start 2026-10-13 14:48:35 EDT load { 13.33 14.16 14.40 }; exit 0 wall 306 end 2026-10-13 14:53:41 EDT load { 13.28 13.58 14.04 }
- scheduling: nice -n 10; OMP/OPENBLAS/MKL/VECLIB/RAYON/JULIA threads = 4
- machine: Apple M3 Ultra, 28 cores, 96 GB; Darwin 25.2.0 arm64
- python: Python 3.14.6 (/Users/petercotton/github/winning-bench-oct13/.venv); julia: julia version 1.12.6
- data: fishing.csv, sha256 in fishing_csv.sha256 (same file as ../2026-10-13)
- cloud_error.py / thetas.json / cloud_error.txt: quadrature error of the sharp-branch clouds at the three fitted points (analysis in the PR that added this directory), run after the benchmark with `PYTHONPATH=. python bench_results/fishing/2026-10-13-benchfix/cloud_error.py bench_results/fishing/2026-10-13-benchfix/py.json bench_results/fishing/2026-10-13-benchfix/thetas.json`

## pip freeze

```

```
