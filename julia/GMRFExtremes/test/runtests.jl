# Closed forms, a Monte Carlo referee (testing only, per the house
# rule), the measured laws of the winning repo's tridiagonal track,
# and the refusal contract.
#   julia --project=julia/GMRFExtremes julia/GMRFExtremes/test/runtests.jl
using Test
using Random: Xoshiro
using SparseArrays: spdiagm

using GMRFExtremes
const GE = GMRFExtremes

function stationary_ar1(phi, n; mu = 0.0)
    GaussMarkovChain(fill(mu, n), 1.0, fill(phi, n - 1),
                     fill(sqrt(1 - phi^2), n - 1))
end

function simulate(c::GaussMarkovChain, m, rng)
    n = length(c)
    X = zeros(m, n)
    X[:, 1] .= c.mu[1] .+ c.sd0 .* randn(rng, m)
    for t in 1:(n - 1)
        X[:, t + 1] .= c.mu[t + 1] .+ c.phi[t] .* (X[:, t] .- c.mu[t]) .+
                       c.s[t] .* randn(rng, m)
    end
    return X
end

@testset "iid closed form" begin
    c = stationary_ar1(0.0, 12)
    for u in (-0.5, 0.5, 1.5, 2.5)
        @test abs(max_cdf(c, u) - GE.ndtr(u)^12) < 2e-4
    end
end

@testset "chain from tridiagonal precision reproduces the chain" begin
    phi, n = 0.8, 15
    # stationary AR(1) precision, unit marginal variance
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ (1 - phi^2)
    o = fill(-phi / (1 - phi^2), n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    c2 = chain_from_precision(zeros(n), Q)
    c1 = stationary_ar1(phi, n)
    for u in (0.0, 1.0, 2.0)
        @test abs(max_cdf(c1, u) - max_cdf(c2, u)) < 1e-9
    end
    P1 = argmax_marginals(c1)
    P2 = argmax_marginals(c2)
    @test maximum(abs.(P1 .- P2)) < 1e-9
end

@testset "precision route preserves index order" begin
    # a stationary chain's argmax law is symmetric, so it cannot detect
    # an orientation flip. Drift breaks the symmetry and pins it.
    n = 12
    phi, s2 = 0.7, 0.5
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ s2
    o = fill(-phi / s2, n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    mu = collect(0.25 .* (1:n))            # rising: late indices favoured
    c = chain_from_precision(mu, Q)
    P = argmax_marginals(c)
    @test P[n] > 5 * P[1]
    @test argmax(P) == n
end

@testset "Monte Carlo referee" begin
    rng = Xoshiro(1)
    mu = [0.0, 0.3, 0.6, 0.4, 0.0, -0.3, -0.1, 0.2, 0.5, 0.1]
    c = GaussMarkovChain(mu, 0.8, fill(0.7, 9),
                         vcat(fill(0.6, 4), fill(0.9, 5)))
    X = simulate(c, 400_000, rng)
    mx = vec(maximum(X, dims = 2))
    for u in (0.5, 1.5, 2.5)
        @test abs(max_cdf(c, u) - (sum(mx .<= u) / length(mx))) < 4e-3
    end
    @test abs(expected_max(c) - (sum(mx) / length(mx))) < 5e-3
    # argmax marginals
    am = argmax_marginals(c)
    counts = zeros(10)
    for r in 1:size(X, 1)
        counts[argmax(@view X[r, :])] += 1
    end
    @test maximum(abs.(am .- counts ./ size(X, 1))) < 5e-3
    # first passage at u = 1.0
    fp = first_passage(c, 1.0)
    @test abs(sum(fp) - 1) < 1e-9
    hit = [findfirst(>(1.0), @view X[r, :]) for r in 1:200_000]
    for t in (1, 3, 7)
        emp = sum(h -> h !== nothing && h == t, hit) / 200_000
        @test abs(fp[t] - emp) < 5e-3
    end
end

@testset "the measured laws of the tridiagonal track" begin
    # exp2: positive correlation piles the argmax onto the BOUNDARY --
    # measured ends/middle ~ 2.6 at phi = 0.9, n = 20
    P = argmax_marginals(stationary_ar1(0.9, 20); points = 400)
    ratio = (P[1] + P[end]) / 2 / P[10]
    @test 2.0 < ratio < 3.4
    @test abs(P[1] - P[end]) < 0.01          # exchange symmetry
    # exp3: the random walk's discrete arcsine U-shape
    n = 40
    w = GaussMarkovChain(zeros(n), 1.0, fill(1.0, n - 1),
                         fill(1.0, n - 1))
    Pw = argmax_marginals(w; points = 400)
    @test Pw[1] > 4 * Pw[div(n, 2)]
    @test Pw[end] > 4 * Pw[div(n, 2)]
end

@testset "refusal contract" begin
    n = 8
    Q = spdiagm(-2 => fill(0.1, n - 2), -1 => fill(-0.4, n - 1),
                0 => fill(1.5, n), 1 => fill(-0.4, n - 1),
                2 => fill(0.1, n - 2))
    @test_throws ErrorException chain_from_precision(zeros(n), Q)
end

@testset "GaussianMarkovRandomFields dispatch extension" begin
    # `using GaussianMarkovRandomFields` loads GMRFDispatchExt, so the
    # GMRF methods below are the extension's own, not local glue
    using GaussianMarkovRandomFields
    GMF = GaussianMarkovRandomFields
    phi, n = 0.8, 12
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ (1 - phi^2)
    o = fill(-phi / (1 - phi^2), n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    g = GMF.GMRF(zeros(n), Q)
    ref = stationary_ar1(phi, n)
    for u in (0.5, 1.5)
        @test abs(max_cdf(g, u) - max_cdf(ref, u)) < 1e-8
    end
    am = argmax_marginals(g)
    @test abs(sum(am) - 1) < 1e-9
    @test abs(am[1] - am[end]) < 0.01
end

println("all GMRFExtremes tests passed")
