# Where simulation stops answering. P(max <= u) on a correlated chain
# is an orthant probability, and as u falls it becomes a rare event:
# a direct sampler needs about 1/p draws to see it at all, and the
# GHK family that computes such orthants degrades in the same regime
# (Ridgway 2016). The restricted transfer operator returns it at
# FIXED cost whatever p is. This measures both.
#   julia run_tail.jl
include(joinpath(@__DIR__, "..", "..", "..", "julia", "GMRFExtremes",
                 "src", "GMRFExtremes.jl"))
using .GMRFExtremes
using Random: Xoshiro
using Printf

n, phi = 50, 0.9
c = GaussMarkovChain(zeros(n), 1.0, fill(phi, n - 1),
                     fill(sqrt(1 - phi^2), n - 1))

function mc_maxcdf(u, m, seed)
    rng = Xoshiro(seed)
    sd = sqrt(1 - phi^2)
    hits = 0
    for _ in 1:m
        x = randn(rng)
        mx = x
        for _ in 2:n
            x = phi * x + sd * randn(rng)
            mx = max(mx, x)
        end
        mx <= u && (hits += 1)
    end
    return hits / m, hits
end

M = 1_000_000
println("stationary AR(1), n = $n, phi = $phi;  P(max <= u)")
println(rpad("u", 7), rpad("exact (L=400)", 16), rpad("exact (L=800)", 16),
        rpad("MC 1e6", 14), rpad("hits", 7), "draws for 10% rel. err")
for u in (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0)
    e4 = max_cdf(c, u; points = 400)
    e8 = max_cdf(c, u; points = 800)
    mc, hits = mc_maxcdf(u, M, 20)
    need = e8 > 0 ? 100 / e8 : Inf                 # rel err = 1/sqrt(Np)
    @printf("%-7.1f%-16.3e%-16.3e%-14.3e%-7d%.1e\n", u, e4, e8, mc, hits, need)
end

t0 = time()
for u in (0.0, -1.0, -2.0, -3.0)
    max_cdf(c, u; points = 400)
end
@printf("\nexact: %.0f ms per threshold, independent of p\n",
        (time() - t0) / 4 * 1000)
