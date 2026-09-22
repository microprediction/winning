#!/bin/zsh
# Resume n=1e5 to a 2^17 truth, Sobol 2^15 and hybrid R=2048; everything already priced is reused from state_largen.
cd /Users/petercotton/github/winning
export PYTHONPATH=.
python3 -u research/factor_ghk/hybrid_a_largen.py --n 100000 --rank 3 --D 0.05 --R 32 128 512 2048 --sobol-m 7 9 11 13 15 --truth-m 17 --paths 100000 --workers 24 \
   --csv research/factor_ghk/runs/largen_n100000.csv 2>&1 | grep --line-buffered -v "Warning\|x, points\|eng.random" >> research/factor_ghk/runs/largen_n100000.log
echo RESUME DONE
