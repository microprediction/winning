#!/bin/zsh
# Hybrid A at large n. n=1000 first on all workers, then n=1e4 and n=1e5 side by side.
cd /Users/petercotton/github/winning
export PYTHONPATH=.
R=research/factor_ghk/runs
run() { python3 -u research/factor_ghk/hybrid_a_largen.py "$@" 2>&1 | grep --line-buffered -v "Warning\|x, points" ; }
run --n 1000   --rank 3 --D 0.05 --R 32 128 512 2048 --sobol-m 7 9 11 13 15 --truth-m 18 --paths 3000000 --ghk 256 1024 4096 --workers 24 --csv $R/largen_n1000.csv > $R/largen_n1000.log
run --n 10000  --rank 3 --D 0.05 --R 32 128 512 2048 --sobol-m 7 9 11 13    --truth-m 17 --paths 300000 --ghk 256 --ghk-max-n 10000 --ghk-targets 2 --workers 12 --csv $R/largen_n10000.csv > $R/largen_n10000.log &
run --n 100000 --rank 3 --D 0.05 --R 32 128 512      --sobol-m 7 9 11 13    --truth-m 15 --paths 100000 --workers 12 --csv $R/largen_n100000.csv > $R/largen_n100000.log &
wait
echo LARGEN DONE
