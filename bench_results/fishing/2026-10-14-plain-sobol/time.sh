#!/bin/zsh
# usage: time.sh <main tree> <branch tree>: 3 alternating rounds of the
# python exact fit and the julia exact fit (ghk skipped), wall seconds
D=${0:A:h}
CSV=/private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad/fishing_sobol-sym.csv
PY=/Users/petercotton/github/winning-bench-oct13/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 JULIA_NUM_THREADS=4
for round in 1 2 3; do for G in main branch; do
  T=$1; [ $G = branch ] && T=$2
  cd $T
  PYTHONPATH=$T nice -n 10 $PY julia/MultinomialProbit/benchmark/fishing_python.py $CSV /private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad/sobol-sym_t_$G.json | sed "s/^/$round $G /"
  nice -n 10 julia --project=julia/MultinomialProbit -e "
    include(\"julia/MultinomialProbit/src/MultinomialProbit.jl\"); using .MultinomialProbit
    include(\"julia/MultinomialProbit/test/minijson.jl\")
    inp = parse_json(read(\"$D/py_$G.json\", String)); Xl = inp[\"X\"]
    T = length(Xl); J = length(Xl[1]); p = length(Xl[1][1]); X = zeros(T, J, p)
    for t in 1:T, j in 1:J, c in 1:p; X[t, j, c] = Xl[t][j][c]; end
    ch = Int.(inp[\"choice\"]) .+ 1
    fit!(MNProbit(X[1:50,:,:], ch[1:50]; intercepts = true, r = 2))
    m = MNProbit(X, ch; intercepts = true, r = 2); t0 = time(); fit!(m)
    println(\"$round $G julia exact \", round(m.loglik, digits = 2), \" in \", round(time() - t0, digits = 1), \"s\")"
done; done
