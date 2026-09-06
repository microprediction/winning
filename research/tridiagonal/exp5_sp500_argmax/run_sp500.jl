# The day of the market's yearly high: the drifted random walk's
# argmax marginals (GMRFExtremes, exact) against 98 years of S&P 500
# closes, with the driftless discrete-arcsine law as the null. The
# engine's claim in the win-nodes paper: the drifted case has no
# closed form and the transfer operator computes it anyway.
#   julia run_sp500.jl sp500_years.json
include(joinpath(@__DIR__, "..", "..", "..", "julia", "GMRFExtremes",
                 "src", "GMRFExtremes.jl"))
using .GMRFExtremes
include(joinpath(@__DIR__, "..", "..", "..", "julia",
                 "MultinomialProbit", "test", "minijson.jl"))

inp = parse_json(read(ARGS[1], String))
mu_d = Float64(inp["mu"])
sd_d = Float64(inp["sd"])
fracs = [Float64(y["frac"]) for y in inp["per_year"]]
ny = length(fracs)

# 63 four-day blocks in standardized units: the block containing the
# yearly high IS the block of the daily argmax, the law is scale-free
# (only drift-per-step delta = 4 mu / (2 sigma) enters), and the grid
# can then resolve the step scale (dx = step sd / 4), which daily
# resolution at n = 252 cannot afford
n = 63
delta = 4 * mu_d / (2 * sd_d)
D = 10                                     # deciles
function decile_law(drift)
    c = GaussMarkovChain(collect(drift .* (1:n)), 1.0,
                         fill(1.0, n - 1), fill(1.0, n - 1))
    P = argmax_marginals(c; points = 520)
    agg = zeros(D)
    for t in 1:n
        agg[min(D, 1 + floor(Int, (t - 0.5) / n * D))] += P[t]
    end
    return agg
end

drifted = decile_law(delta)
arcsine = decile_law(0.0)
emp = zeros(D)
for f in fracs
    emp[min(D, 1 + floor(Int, f * D))] += 1.0 / ny
end

println("decile  empirical  drifted  arcsine(driftless)")
for d in 1:D
    println(rpad(d, 7), rpad(round(emp[d], digits = 3), 11),
            rpad(round(drifted[d], digits = 3), 9),
            round(arcsine[d], digits = 3))
end
ll(p) = sum(log(max(p[min(D, 1 + floor(Int, f * D))], 1e-12))
            for f in fracs)
println("log-likelihood over ", ny, " years: drifted ",
        round(ll(drifted), digits = 2), "  arcsine ",
        round(ll(arcsine), digits = 2), "  uniform ",
        round(ny * log(1 / D), digits = 2))
println("last-decile share: empirical ", round(emp[D], digits = 3),
        "  drifted ", round(drifted[D], digits = 3),
        "  arcsine ", round(arcsine[D], digits = 3))
