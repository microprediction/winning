#!/bin/zsh
# usage: run.sh <tree> <tag>   (from anywhere; writes into this dir)
D=${0:A:h}; T=$1; G=$2
CSV=/private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad/fishing_sobol-sym.csv
PY=/Users/petercotton/github/winning-bench-oct13/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 JULIA_NUM_THREADS=4
cd $T
echo "== $G $(git rev-parse HEAD) start $(date) load $(sysctl -n vm.loadavg)"
PYTHONPATH=$T nice -n 10 $PY julia/MultinomialProbit/benchmark/fishing_python.py $CSV $D/py_$G.json
nice -n 10 julia --project=julia/MultinomialProbit julia/MultinomialProbit/benchmark/fishing_julia.jl $D/py_$G.json $D/jl_$G.json
PYTHONPATH=$T nice -n 10 $PY julia/MultinomialProbit/benchmark/fishing_referee.py $D/py_$G.json $D/jl_$G.json
PYTHONPATH=$T nice -n 10 $PY $D/score.py $D/py_$G.json $D/jl_$G.json
echo "== $G end $(date) load $(sysctl -n vm.loadavg)"
