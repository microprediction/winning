# Parity with the Python reference plus a few intrinsic invariants.
# parity/gen_vectors.py writes the vectors; parity/check.jl runs the
# same scenarios standalone against the source tree.
#   julia -e 'using Pkg; Pkg.develop(path = "julia/winning"); Pkg.test("winning")'
using Test
using winning

include(joinpath(@__DIR__, "minijson.jl"))
include(joinpath(@__DIR__, "parity.jl"))

const VECTORS = joinpath(@__DIR__, "..", "..", "..", "parity", "vectors.json")

@testset "invariants" begin
    mu = [0.3, -0.1, 0.0, 0.4]
    D = ones(4)
    p = race_probabilities(mu; D = D, points = 257)
    @test length(p) == 4
    @test all(p .> 0)
    @test isapprox(sum(p), 1.0; atol = 1e-8)
    perm = [3, 1, 4, 2]
    @test isapprox(race_probabilities(mu[perm]; D = D, points = 257), p[perm];
                   atol = 1e-10)
    @test isapprox(race_probabilities(mu .+ 0.7; D = D, points = 257), p;
                   atol = 1e-8)
end

@testset "parity with the python reference" begin
    isfile(VECTORS) || error("parity/vectors.json not found; the tests run " *
                             "from the winning repository checkout " *
                             "(python parity/gen_vectors.py writes it)")
    fails, skipped = run_parity(VECTORS)
    @test fails == 0
    @test "independent_normal" ∉ skipped
end

# --- caller-supplied factor nodes carry the loadings' rank ------------
#
# _race_setup took the caller's F and W verbatim. An F with the wrong
# number of ROWS was accepted and priced a different quadrature outright
# -- [0.646, 0.123, 0.231, 0.0005] where the right answer is
# [0.382, 0.301, 0.137, 0.181] -- and an extra weight was silently
# ignored. The two spellings that did fail failed with a
# DimensionMismatch or a BoundsError from somewhere inside, naming
# nothing the caller passed. This is #290, which was filed against the
# browser; julia had its own copy.
@testset "factor nodes and weights are checked at the door" begin
    mu = [-0.6, -0.2, 0.15, 0.7]
    V = [1.2 -0.7; -0.4 1.1; 0.6 0.9; -1.0 -0.5]
    D = [0.5, 0.8, 0.6, 0.9]
    F = [-1.0 -1.0; -1.0 1.0; 1.0 -1.0; 1.0 1.0]
    W = fill(0.25, 4)

    p0 = race_probabilities(mu; V = V, D = D, F = F, W = W)
    @test isapprox(sum(p0), 1.0; atol = 1e-12)

    # W and c*W are the same factor law
    for c in (1e-6, 0.1, 10.0, 1e6)
        @test maximum(abs.(race_probabilities(mu; V = V, D = D, F = F,
                                              W = W .* c) .- p0)) < 1e-15
    end
    # and so is an unnormalised spelling of it
    @test maximum(abs.(race_probabilities(mu; V = V, D = D, F = F,
                                          W = ones(4)) .- p0)) < 1e-15

    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F[:, 1:1], W = W)
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F[1:2, :], W = W)
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F, W = W[1:2])
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F, W = vcat(W, 0.9))
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F, W = [0.5, -0.5, 0.5, 0.5])
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = F, W = zeros(4))
    @test_throws ArgumentError race_probabilities(mu; V = V, D = D,
                                                  F = zeros(0, 2), W = Float64[])

    # the internally built nodes are untouched
    @test isapprox(sum(race_probabilities(mu; V = V, D = D)), 1.0; atol = 1e-12)
end
