# Run manifest: Laplace inverse micro-benchmark (new; no standing suite covers it)

- purpose: measure the cost of #707 (Newton own-slopes for kinked bases from the split lattice: one extra forward pass per inverse iteration)
- script: bench_results/laplace_inverse/2026-10-13/laplace_inverse_bench.py (seeded self-generated targets; best of 3 per case; mu_err = max |centred mu_hat - mu_true|)
- commands (run twice, alternating, from the worktree root):
  - `PYTHONPATH=<worktree at f1a0e68e> nice -n 19 .venv/bin/python bench_results/laplace_inverse/2026-10-13/laplace_inverse_bench.py pre707_f1a0e68e >> bench_results/laplace_inverse/2026-10-13/records.jsonl`
  - `PYTHONPATH=$PWD nice -n 19 .venv/bin/python bench_results/laplace_inverse/2026-10-13/laplace_inverse_bench.py main_a7f7d9fd >> bench_results/laplace_inverse/2026-10-13/records.jsonl`
- code under test: f1a0e68e (main just before the #707 merge) vs a7f7d9fd (origin/main). NOTE: the "sha" field of the main_a7f7d9fd rows reads 29acbdc6, the bench-oct13 results commit checked out at the time; its code is a7f7d9fd unchanged (that commit only added files under bench_results/).
- fastrace 0.3.0 built from rust/fastrace at a7f7d9fd for both (rust/ is unchanged between the two SHAs)
- run: 2026-10-13 13:24 EDT, load average { 31.09 33.48 33.32 }, threads capped at 4, nice 19
- machine: Apple M3 Ultra, 28 cores; Darwin 25.2.0 arm64; Python 3.14.6; numpy 2.5.1, scipy 1.18.0
