include(ARGS[1]); using .MultinomialProbit
include("/Users/petercotton/github/winning-fix6-ports6/julia/MultinomialProbit/test/minijson.jl")
inp = parse_json(read(ARGS[2], String)); Xl = inp["X"]
T = length(Xl); J = length(Xl[1]); p = length(Xl[1][1]); X = zeros(T, J, p)
for t in 1:T, j in 1:J, c in 1:p; X[t, j, c] = Xl[t][j][c]; end
ch = Int.(inp["choice"]) .+ 1
fit!(MNProbit(X[1:50,:,:], ch[1:50]; intercepts = true, r = 2))
m = MNProbit(X, ch; intercepts = true, r = 2); t0 = time(); fit!(m)
th = m.theta
println(ARGS[3], " julia exact ", round(m.loglik, digits = 2), " in ", round(time() - t0, digits = 1), "s")
open(ARGS[3] * ".theta", "w") do io; println(io, join(th, ",")); end
