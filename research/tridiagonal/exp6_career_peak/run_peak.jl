# When was the player at their peak? The query side: argmax marginals
# on the Laplace posterior over the ability path, which is a
# Gauss-Markov chain because the win evidence is per-year and the
# prior is a random walk. Monte Carlo referees the exact pass.
#   julia run_peak.jl tennis_paths.json
include(joinpath(@__DIR__, "..", "..", "..", "julia", "GMRFExtremes",
                 "src", "GMRFExtremes.jl"))
using .GMRFExtremes
include(joinpath(@__DIR__, "..", "..", "..", "julia",
                 "MultinomialProbit", "test", "minijson.jl"))
using LinearAlgebra: SymTridiagonal, cholesky, Symmetric
using Random: Xoshiro

inp = parse_json(read(ARGS[1], String))

function credible_set(P, level)
    ord = sortperm(P, rev = true)
    acc = 0.0
    out = Int[]
    for i in ord
        push!(out, i)
        acc += P[i]
        acc >= level && break
    end
    return sort(out)
end

function mc_argmax(mu, Q, m, seed)
    n = length(mu)
    C = cholesky(Symmetric(Matrix(Q)))
    counts = zeros(n)
    rng = Xoshiro(seed)
    for _ in 1:m
        x = mu .+ (C.U \ randn(rng, n))     # N(mu, Q^{-1})
        counts[argmax(x)] += 1
    end
    return counts ./ m
end

println(rpad("player", 17), rpad("MAP peak", 10), rpad("P(peak)", 9),
        rpad("most likely", 13), "80% credible set")
rows = []
for pl in inp["players"]
    mu = Float64.(pl["mean"])
    n = length(mu)
    d = Float64.(pl["Q_diag"])
    o = Float64.(pl["Q_off"])
    Q = Matrix(SymTridiagonal(d, o))
    c = chain_from_precision(mu, Q)
    P = argmax_marginals(c; points = 400)
    y0 = Int(pl["first"])
    mapk = argmax(mu)
    modek = argmax(P)
    cs = credible_set(P, 0.8)
    ref = mc_argmax(mu, Q, 200_000, 7)
    push!(rows, (pl["name"], y0, mu, P, ref, mapk, modek, cs))
    println(rpad(pl["name"], 17), rpad(y0 + mapk - 1, 10),
            rpad(round(P[mapk], digits = 3), 9),
            rpad(y0 + modek - 1, 13),
            "$(y0 + cs[1] - 1)-$(y0 + cs[end] - 1) ($(length(cs)) yrs)")
end

println()
maxerr = maximum(maximum(abs.(r[4] .- r[5])) for r in rows)
println("Monte Carlo referee (200k posterior draws each): max |exact - MC| = ",
        round(maxerr, digits = 4))

# the sparse-year effect: does match count predict argmax mass at
# equal posterior mean?
println()
for r in rows
    name, y0, mu, P, _, mapk, modek, _ = r
    if mapk != modek
        println(name, ": smoothed-path peak ", y0 + mapk - 1,
                " but most-likely peak ", y0 + modek - 1,
                "  (P = ", round(P[mapk], digits = 3), " vs ",
                round(P[modek], digits = 3), ")")
    end
end
for r in rows
    name, y0, mu, P, _, _, _, _ = r
    if name == "Roger Federer"
        println("\nFederer year-by-year: P(peak year)")
        for t in eachindex(P)
            P[t] > 0.02 && println("  ", y0 + t - 1, "  ",
                                   round(P[t], digits = 3))
        end
    end
end
