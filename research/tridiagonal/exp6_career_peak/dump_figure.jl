# Dump one player's posterior path, its marginal sds and its argmax
# law for the paper figure.
#   julia dump_figure.jl tennis_paths.json fig.json [player]
include(joinpath(@__DIR__, "..", "..", "..", "julia", "GMRFExtremes",
                 "src", "GMRFExtremes.jl"))
using .GMRFExtremes
include(joinpath(@__DIR__, "..", "..", "..", "julia",
                 "MultinomialProbit", "test", "minijson.jl"))
using LinearAlgebra: SymTridiagonal, inv, diag

inp = parse_json(read(ARGS[1], String))
who = length(ARGS) > 2 ? ARGS[3] : "Rafael Nadal"
pl = only(filter(p -> p["name"] == who, inp["players"]))
mu = Float64.(pl["mean"])
Q = Matrix(SymTridiagonal(Float64.(pl["Q_diag"]), Float64.(pl["Q_off"])))
sd = sqrt.(diag(inv(Q)))
Pk = argmax_marginals(chain_from_precision(mu, Q); points = 400)
y0 = Int(pl["first"])
years = collect(y0:(y0 + length(mu) - 1))
open(ARGS[2], "w") do io
    print(io, "{\"years\": [", join(years, ","),
          "], \"mean\": [", join(mu, ","),
          "], \"sd\": [", join(sd, ","),
          "], \"argmax\": [", join(Pk, ","), "]}")
end
println("dumped ", who, ": ", length(years), " seasons")
