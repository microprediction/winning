#!/bin/bash
# usage: fish_at.sh SHA  -> runs python + julia exact Fishing fit at SHA in the bisect worktree
W=/Users/petercotton/github/winning-bench-pre707
S=/private/tmp/claude-501/-Users-petercotton-github-winning/1e9db0b1-8575-4a57-a5bf-81b8e7a714aa/scratchpad
PY=/Users/petercotton/github/winning-bench-oct13/.venv/bin/python
FISH=/Users/petercotton/github/winning-bench-oct13/.venv/data/fishing.csv
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 JULIA_NUM_THREADS=2
git -C $W checkout -q --detach $1 || exit 9
cd $S
py=$(PYTHONPATH=$W nice -n 19 $PY $W/julia/MultinomialProbit/benchmark/fishing_python.py $FISH $S/py_$1.json 2>&1 | tail -1)
jl=$(nice -n 19 julia -e "include(\"$W/julia/MultinomialProbit/src/MultinomialProbit.jl\"); using .MultinomialProbit; include(\"$W/julia/MultinomialProbit/test/minijson.jl\");
inp = parse_json(read(\"$S/py_$1.json\", String)); Xl = inp[\"X\"]; T = length(Xl); J = length(Xl[1]); p = length(Xl[1][1]);
X = zeros(T, J, p); for t in 1:T, j in 1:J, c in 1:p; X[t,j,c] = Xl[t][j][c]; end
m = MNProbit(X, Int.(inp[\"choice\"]) .+ 1; intercepts=true, r=2); t0=time(); fit!(m); println(\"julia exact logLik \", round(m.loglik,digits=2), \" \", round(time()-t0,digits=1), \"s conv \", m.converged)" 2>&1 | grep -E "julia exact|Error" | head -1)
echo "$1 | $py | $jl"
