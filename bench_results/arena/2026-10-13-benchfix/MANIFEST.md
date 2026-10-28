# Run manifest

- suite dir: bench_results/arena/2026-10-13-benchfix (fix3-benchfix: #716 #717 #718)
- command (repo root, PYTHONPATH=.): `python -m winning.bench.runner --out bench_results/arena/2026-10-13-benchfix && python -m winning.bench.leaderboard --out bench_results/arena/2026-10-13-benchfix`
- git SHA (code under test): abf367b616f09cdf43f55fb05e0cc59f66961bcb
- references: ref_*.npy copied byte-for-byte from ../2026-10-13-rerun
- start 2026-10-13 14:53:49 EDT load { 13.18 13.55 14.02 }; exit 0 wall 558 end 2026-10-13 15:03:07 EDT load { 13.16 13.26 13.63 }
- scheduling: nice -n 10, single process, BLAS/rayon threads = 4 (same caps as ../2026-10-13-rerun and ../2026-10-13-baseline-3b9eaa18)
- machine: Apple M3 Ultra, 28 cores; Darwin 25.2.0 arm64; Python 3.14.6; fastrace 0.3.0 from rust/fastrace at a7f7d9fd (rust unchanged)
- CMP_vs_*.md: `python bench_results/arena/2026-10-13-bisect/cmp_arena.py OLD/records.jsonl bench_results/arena/2026-10-13-benchfix/records.jsonl`

## pip freeze

```

```
