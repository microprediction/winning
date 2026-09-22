#!/bin/zsh
# Hybrid A sweep: one process per field, 2 BLAS threads each (LPs are single-threaded anyway).
cd /Users/petercotton/github/winning
export PYTHONPATH=.
run() { python3 -u research/factor_ghk/hybrid_a.py "$@" 2>&1 | grep --line-buffered -v "Warning\|x, points" ; }
for rank in 2 3 4 5; do
  tm=18; [ $rank -ge 3 ] && tm=20
  for D in 0.1 0.05 0.02; do
    run --rank $rank --n 8 --D $D --R 32 128 512 2048 --ghk 128 512 2048 8192 32768 --paths 3000000 --truth-m $tm \
        --csv research/factor_ghk/runs/sweep_rank${rank}_D${D}.csv > research/factor_ghk/runs/sweep_rank${rank}_D${D}.log &
  done
done
for margin in 3 6; do
  run --rank 3 --n 8 --D 0.05 --R 32 128 512 2048 --ghk 128 512 2048 8192 32768 --paths 3000000 --truth-m 20 --margin $margin \
      --csv research/factor_ghk/runs/sweep_rank3_D0.05_margin${margin}.csv > research/factor_ghk/runs/sweep_rank3_D0.05_margin${margin}.log &
done
wait
echo SWEEP DONE
