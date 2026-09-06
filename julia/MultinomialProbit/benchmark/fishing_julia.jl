# Fishing benchmark, Julia side: fit both engines on the arrays the
# python script emitted, write thetas back for the shared referee.
#   julia julia/MultinomialProbit/benchmark/fishing_julia.jl in.json out.json
include(joinpath(@__DIR__, "..", "src", "MultinomialProbit.jl"))
using .MultinomialProbit
include(joinpath(@__DIR__, "..", "test", "minijson.jl"))

inp = parse_json(read(ARGS[1], String))
Xl = inp["X"]
T = length(Xl); J = length(Xl[1]); p = length(Xl[1][1])
X = zeros(T, J, p)
for t in 1:T, j in 1:J, c in 1:p
    X[t, j, c] = Xl[t][j][c]
end
choice = Int.(inp["choice"]) .+ 1

m = MNProbit(X, choice; intercepts = true, r = 2)
t0 = time()
fit!(m)
dt_exact = time() - t0
println("julia exact: logLik(GH) ", round(m.loglik, digits = 2),
        " in ", round(dt_exact, digits = 1), "s, converged ", m.converged)
theta_exact = copy(m.theta)

t0 = time()
fit!(m; method = :ghk, r_draws = 200, maxiter = 60)
dt_ghk = time() - t0
println("julia ghk:   logLik(sim) ", round(m.loglik, digits = 2),
        " in ", round(dt_ghk, digits = 1), "s, converged ", m.converged)

open(ARGS[2], "w") do io
    print(io, "{\"theta_exact\": [", join(theta_exact, ","),
          "], \"theta_ghk\": [", join(m.theta, ","),
          "], \"seconds_exact\": ", dt_exact,
          ", \"seconds_ghk\": ", dt_ghk, "}")
end
