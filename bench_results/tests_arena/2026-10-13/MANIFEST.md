# Run manifest

- suite dir: bench_results/tests_arena/2026-10-13
- command: `/Users/petercotton/github/winning-bench-oct13/.venv/bin/python -m pytest -p no:cacheprovider -q --durations=0 --junitxml=bench_results/tests_arena/2026-10-13/junit.xml tests_arena tests_experiments/test_ghk_benchmark.py tests_research/test_perf.py`
- git SHA (code under test): a7f7d9fdd9873f81677b849455793abf3bf19bd0
- started: 2026-10-13 13:12:20 EDT  finished: 2026-10-13 13:12:27 EDT  exit: 0
- wall seconds: 7
- load average at start: { 32.79 32.61 31.18 } ; at end: { 32.73 32.60 31.18 }
- scheduling: nice -n 19; OMP/OPENBLAS/MKL/VECLIB/RAYON/JULIA threads = 4
- machine: Apple M3 Ultra, 28 cores, 96 GB; Darwin 25.2.0 arm64
- python: Python 3.14.6 (/Users/petercotton/github/winning-bench-oct13/.venv/bin/python)
- winning imported from: /Users/petercotton/github/winning-bench-oct13/winning/__init__.py
- fastrace: 0.3.0 built from rust/fastrace at a7f7d9fdd9873f81677b849455793abf3bf19bd0; rust_active=True
- R: R version 4.6.1 (2026-06-24) mvtnorm 1.4.1
- julia: julia version 1.12.6

## pip freeze

```
contourpy==1.4.0
cycler==0.12.1
fastrace @ file:///Users/petercotton/github/winning-bench-oct13/rust/fastrace
fonttools==4.66.1
iniconfig==2.3.1
kiwisolver==1.5.1
matplotlib==3.11.2
maturin==1.15.0
numpy==2.5.1
openskill==6.2.0
packaging==26.3
pandas==3.0.3
pillow==12.3.0
pluggy==1.6.0
pygments==2.21.0
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
scipy==1.18.0
six==1.17.0
trueskill==0.4.5
```
