#!/bin/bash
# Benchmark rerun on origin/main, 2026-10-13. Launched as
#   nohup nice -n 19 bash bench_results/runs/2026-10-13/run.sh > bench_results/runs/2026-10-13/driver.log 2>&1 &
# from the worktree root. Every suite writes into its own NEW dated directory
# (bench_results/<suite>/2026-10-13/), nothing earlier is overwritten, and each
# suite is committed to the branch as soon as it finishes.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
DAY=2026-10-13
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 JULIA_NUM_THREADS=4
SHA=$(git rev-parse HEAD)

manifest () {  # $1 dir, $2 command text
  {
    echo "# Run manifest"
    echo
    echo "- suite dir: $1"
    echo "- command: \`$2\`"
    echo "- git SHA (code under test): $SHA"
    echo "- started: $START  finished: $(date '+%Y-%m-%d %H:%M:%S %Z')  exit: $RC"
    echo "- wall seconds: $(( $(date +%s) - START_S ))"
    echo "- load average at start: $LOAD0 ; at end: $(sysctl -n vm.loadavg)"
    echo "- scheduling: nice -n 19; OMP/OPENBLAS/MKL/VECLIB/RAYON/JULIA threads = 4"
    echo "- machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB; $(uname -srm)"
    echo "- python: $($PY --version 2>&1) ($PY)"
    echo "- winning imported from: $($PY -c 'import winning;print(winning.__file__)' 2>&1)"
    echo "- fastrace: $($PY -c 'import importlib.metadata as m;print(m.version("fastrace"))' 2>&1) built from rust/fastrace at $SHA; rust_active=$($PY -c 'import winning.rustconfig as r;print(r.rust_active())' 2>&1)"
    echo "- R: $(Rscript -e 'cat(R.version.string, "mvtnorm", as.character(packageVersion("mvtnorm")))' 2>&1)"
    echo "- julia: $(julia --version 2>&1)"
    echo
    echo "## pip freeze"
    echo
    echo '```'
    $PY -m pip freeze 2>/dev/null || uv pip freeze --python "$PY"
    echo '```'
  } > "$1/MANIFEST.md"
}

commit () {  # $1 dir, $2 message
  git add -- "$1" bench_results/runs/$DAY/run.sh
  git commit -q -m "$2" -m "Raw results from bench_results/runs/$DAY/run.sh at $SHA; see $1/MANIFEST.md." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$1" bench_results/runs/$DAY/run.sh \
    && echo "committed $1" || echo "COMMIT FAILED for $1"
}

begin () { START=$(date '+%Y-%m-%d %H:%M:%S %Z'); START_S=$(date +%s); LOAD0=$(sysctl -n vm.loadavg); echo "=== $1 start $START"; }

# 1. Arena: full grid, same seeds, SAME cached references as the committed run
D=bench_results/arena/$DAY; mkdir -p $D
cp -n bench_results/ref_*.npy $D/
begin arena
CMD="$PY -m winning.bench.runner --out $D && $PY -m winning.bench.leaderboard --out $D"
$PY -m winning.bench.runner --out $D > $D/stdout.txt 2>&1; RC=$?
$PY -m winning.bench.leaderboard --out $D >> $D/stdout.txt 2>&1 || RC=$((RC+100))
manifest $D "$CMD"; commit $D "Arena benchmark rerun on main ($DAY): raw records + leaderboard"

# 2. Seasons (stdout tables; SEASONS.md was produced from season_ranked)
D=bench_results/seasons/$DAY; mkdir -p $D
begin seasons
RC=0
for mod in season_ranked season_trueskill season; do
  /usr/bin/time -p $PY -m winning.bench.$mod > $D/$mod.txt 2> $D/$mod.time.txt || RC=$((RC+1))
done
manifest $D "for mod in season_ranked season_trueskill season: $PY -m winning.bench.\$mod > $D/\$mod.txt"
commit $D "Season benchmarks rerun on main ($DAY)"

# 3. Arena anchors + benchmark-flavoured tests, with durations
D=bench_results/tests_arena/$DAY; mkdir -p $D
begin tests
CMD="$PY -m pytest -p no:cacheprovider -q --durations=0 --junitxml=$D/junit.xml tests_arena tests_experiments/test_ghk_benchmark.py tests_research/test_perf.py"
$PY -m pytest -p no:cacheprovider -q --durations=0 --junitxml=$D/junit.xml \
    tests_arena tests_experiments/test_ghk_benchmark.py tests_research/test_perf.py > $D/stdout.txt 2>&1; RC=$?
manifest $D "$CMD"; commit $D "Arena anchor tests and perf tests rerun on main ($DAY)"

# 4. Fishing cross-language MNP benchmark (J=4, rank 2)
D=bench_results/fishing/$DAY; mkdir -p $D
begin fishing
FISH="$ROOT/.venv/data/fishing.csv"   # mlogit 1.1-1 data/Fishing.rda -> write.csv; sha256 recorded below
shasum -a 256 "$FISH" > $D/fishing_csv.sha256
B=julia/MultinomialProbit/benchmark
CMD="$PY $B/fishing_python.py fishing.csv $D/py.json; julia $B/fishing_julia.jl $D/py.json $D/jl.json; $PY $B/fishing_referee.py $D/py.json $D/jl.json"
RC=0
$PY $B/fishing_python.py "$FISH" $D/py.json > $D/stdout.txt 2>&1 || RC=1
julia $B/fishing_julia.jl $D/py.json $D/jl.json >> $D/stdout.txt 2>&1 || RC=$((RC+2))
$PY $B/fishing_referee.py $D/py.json $D/jl.json >> $D/stdout.txt 2>&1 || RC=$((RC+4))
manifest $D "$CMD"; commit $D "Fishing cross-language MNP benchmark rerun on main ($DAY)"

# 5. mvtnorm (R) on the arena problems -- control: no winning numerics in the path
D=bench_results/mvtnorm/$DAY; mkdir -p $D
cp -n bench_results/ref_*.npy $D/
begin mvtnorm
CMD="$PY -m winning.bench.mvtnorm_check --out $D"
$PY -m winning.bench.mvtnorm_check --out $D > $D/stdout.txt 2>&1; RC=$?
manifest $D "$CMD"; commit $D "mvtnorm control benchmark rerun ($DAY)"

echo "=== all done $(date)"
