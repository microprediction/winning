#!/bin/bash
# Arena rerun after the runner serialisation fix (9b249304). The first
# attempt (bench_results/arena/2026-10-13/) crashed at the first ghk record
# and is kept as-is. Launched as
#   nohup nice -n 19 bash bench_results/runs/2026-10-13/run_arena2.sh > bench_results/runs/2026-10-13/driver_arena2.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
DAY=2026-10-13; PY="$ROOT/.venv/bin/python"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4
SHA=$(git rev-parse HEAD)
D=bench_results/arena/$DAY-rerun; mkdir -p $D; cp -n bench_results/ref_*.npy $D/
START=$(date '+%Y-%m-%d %H:%M:%S %Z'); S0=$(date +%s); L0=$(sysctl -n vm.loadavg)
$PY -m winning.bench.runner --out $D > $D/stdout.txt 2>&1; RC=$?
$PY -m winning.bench.leaderboard --out $D >> $D/stdout.txt 2>&1 || RC=$((RC+100))
{
echo "# Run manifest"; echo
echo "- command: \`$PY -m winning.bench.runner --out $D && $PY -m winning.bench.leaderboard --out $D\`"
echo "- git SHA: $SHA (origin/main a7f7d9fd + bench_results commits + the runner JSON fix 9b249304; no numerical code changed)"
echo "- references: ref_*.npy copied byte-for-byte from bench_results/ (the 2026-08-15 cached 2e6-draw direct-MC references)"
echo "- started: $START finished: $(date '+%Y-%m-%d %H:%M:%S %Z') exit: $RC wall seconds: $(( $(date +%s) - S0 ))"
echo "- load average at start: $L0 ; at end: $(sysctl -n vm.loadavg) (other users' nice-19 jobs were running on the machine)"
echo "- scheduling: nice -n 19, single process, BLAS/rayon threads = 4"
echo "- machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores; $(uname -srm)"
echo "- python: $($PY --version 2>&1); fastrace $($PY -c 'import importlib.metadata as m;print(m.version("fastrace"))') built from rust/fastrace at a7f7d9fd; rust_active=$($PY -c 'import winning.rustconfig as r;print(r.rust_active())')"
echo; echo "## pip freeze"; echo '```'; uv pip freeze --python "$PY"; echo '```'
} > $D/MANIFEST.md
git add -- $D bench_results/runs/$DAY/run_arena2.sh
git commit -q -m "Arena benchmark rerun on main ($DAY), after the runner fix: raw records + leaderboard" -m "See $D/MANIFEST.md." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $D bench_results/runs/$DAY/run_arena2.sh && echo "committed $D" || echo "COMMIT FAILED"
echo "=== arena2 done $(date)"
