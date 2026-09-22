#!/bin/zsh
cd /Users/petercotton/github/winning; export PYTHONPATH=.
R=research/factor_ghk/runs
run() { python3 -u research/factor_ghk/nodes_b.py "$@" 2>&1 | grep --line-buffered -v "Warning\|x, points" ; }
run --exp prune --n 100000 --m 11 13 15 --csv $R/nodes_b_prune.csv > $R/nodes_b_prune_n100000.log &
run --exp prune --n 10000  --m 11 13 15 --csv $R/nodes_b_prune.csv > $R/nodes_b_prune_n10000.log &
run --exp small --csv $R/nodes_b_small.csv > $R/nodes_b_small.log &
run --exp largen --n 1000 --csv $R/nodes_b_n1000.csv > $R/nodes_b_n1000.log &
wait; echo NODES_B DONE
