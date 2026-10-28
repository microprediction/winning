# Run manifest

- command: `cd <checkout of 3b9eaa18> && PYTHONPATH=. python -m winning.bench.runner` (bench_results symlinked here)
- code under test: 3b9eaa18 (2026-08-15, the commit that produced bench_results/records.jsonl), fastrace 0.1.0 built from its rust/fastrace
- started: 2026-10-13 13:51:24 EDT finished: 2026-10-13 14:00:46 EDT exit: 0 wall seconds: 562
- load average at start: { 29.05 32.12 32.66 } ; at end: { 14.69 17.83 24.64 }
- scheduling: nice -n 19, single process, BLAS/rayon threads = 4 (identical to the 2026-10-13-rerun of main)
- machine: Apple M3 Ultra, 28 cores; Darwin 25.2.0 arm64; Python 3.14.6

## pip freeze
```
fastrace @ file:///private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad/wheels-old/fastrace-0.1.0-cp314-cp314-macosx_11_0_arm64.whl
maturin==1.15.0
numpy==2.5.1
pandas==3.0.3
python-dateutil==2.9.0.post0
scipy==1.18.0
six==1.17.0
trueskill==0.4.5
```
