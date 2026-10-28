# Run manifest

- command: `/Users/petercotton/github/winning-bench-oct13/.venv/bin/python -m winning.bench.runner --out bench_results/arena/2026-10-13-rerun && /Users/petercotton/github/winning-bench-oct13/.venv/bin/python -m winning.bench.leaderboard --out bench_results/arena/2026-10-13-rerun`
- git SHA: 9b249304c86fdd8b13d003e4bcd85b05bf4b3974 (origin/main a7f7d9fd + bench_results commits + the runner JSON fix 9b249304; no numerical code changed)
- references: ref_*.npy copied byte-for-byte from bench_results/ (the 2026-08-15 cached 2e6-draw direct-MC references)
- started: 2026-10-13 13:39:25 EDT finished: 2026-10-13 13:49:17 EDT exit: 0 wall seconds: 592
- load average at start: { 30.45 30.66 31.65 } ; at end: { 29.81 34.06 33.36 } (other users' nice-19 jobs were running on the machine)
- scheduling: nice -n 19, single process, BLAS/rayon threads = 4
- machine: Apple M3 Ultra, 28 cores; Darwin 25.2.0 arm64
- python: Python 3.14.6; fastrace 0.3.0 built from rust/fastrace at a7f7d9fd; rust_active=True

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
