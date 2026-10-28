#!/bin/bash
# Same-machine, same-environment baseline: the arena exactly as committed on
# 2026-08-15 (3b9eaa18, the commit that wrote bench_results/records.jsonl),
# run today with the same caps as the main rerun, to separate code changes
# from environment (thread caps, load, python/numpy) in the old-vs-new table.
# The 3b9eaa18 runner writes to <checkout>/bench_results, which is symlinked
# to bench_results/arena/2026-10-13-baseline-3b9eaa18 in the bench-oct13 worktree so the raw records land in the repo directly.
# fastrace at 3b9eaa18 (pyo3 0.22) built with PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 for python 3.14.
#   nohup nice -n 19 bash bench_results/runs/2026-10-13/run_arena_baseline.sh > bench_results/runs/2026-10-13/driver_arena_baseline.log 2>&1 &
set -u
cd /Users/petercotton/github/winning-bench-oct13
export PYTHONPATH=/Users/petercotton/github/winning-bench-pre707 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4
PY=/private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad/venv-old/bin/python
START=$(date '+%Y-%m-%d %H:%M:%S %Z'); S0=$(date +%s); L0=$(sysctl -n vm.loadavg)
(cd /Users/petercotton/github/winning-bench-pre707 && $PY -m winning.bench.runner > /Users/petercotton/github/winning-bench-oct13/bench_results/arena/2026-10-13-baseline-3b9eaa18/stdout.txt 2>&1); RC=$?
{
echo "# Run manifest"; echo
echo "- command: \`cd <checkout of 3b9eaa18> && PYTHONPATH=. python -m winning.bench.runner\` (bench_results symlinked here)"
echo "- code under test: 3b9eaa18 (2026-08-15, the commit that produced bench_results/records.jsonl), fastrace 0.1.0 built from its rust/fastrace"
echo "- started: $START finished: $(date '+%Y-%m-%d %H:%M:%S %Z') exit: $RC wall seconds: $(( $(date +%s) - S0 ))"
echo "- load average at start: $L0 ; at end: $(sysctl -n vm.loadavg)"
echo "- scheduling: nice -n 19, single process, BLAS/rayon threads = 4 (identical to the 2026-10-13-rerun of main)"
echo "- machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores; $(uname -srm); $($PY --version)"
echo; echo "## pip freeze"; echo '```'; uv pip freeze --python "$PY"; echo '```'
} > bench_results/arena/2026-10-13-baseline-3b9eaa18/MANIFEST.md
git add -- bench_results/arena/2026-10-13-baseline-3b9eaa18 bench_results/runs/2026-10-13/run_arena_baseline.sh
git commit -q -m "Arena same-machine baseline: the 2026-08-15 code (3b9eaa18) rerun today under the same caps" -m "Separates code changes from environment in the old-vs-new arena comparison." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- bench_results/arena/2026-10-13-baseline-3b9eaa18 bench_results/runs/2026-10-13/run_arena_baseline.sh && echo "committed bench_results/arena/2026-10-13-baseline-3b9eaa18" || echo "COMMIT FAILED"
echo "=== baseline done $(date)"
