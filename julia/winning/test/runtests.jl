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
