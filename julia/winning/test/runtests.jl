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

# --- julia must refuse what the other ports refuse --------------------
#
# Found by parity/check_divergence.py, which runs the same malformed
# inputs through python, R, julia and the browser and fails when they
# disagree. julia returned NaN for a zero idiosyncratic variance where
# the other three refuse, and refused the scalar V that as_loadings
# documents and python and the browser accept.
@testset "the refusals match the other ports" begin
    mu = [0.0, 0.3, -0.2, 0.5]
    want = race_probabilities(mu; D = ones(4))

    # a scalar D is the same variance for everyone
    @test race_probabilities(mu; D = 1.0) ≈ want

    # a zero variance is a contestant the lattice cannot represent
    @test_throws ArgumentError race_probabilities(mu; D = [1.0, 0.0, 1.0, 1.0])
    @test_throws ArgumentError race_probabilities(mu; D = [1.0, 1.0])
    @test_throws ArgumentError race_probabilities(mu; D = [1.0, -1.0, 1.0, 1.0])
    @test_throws ArgumentError race_probabilities(mu; D = [1.0, NaN, 1.0, 1.0])

    # a SCALAR V is the same loading for everyone, and a common loading
    # is gauge-fixed away, so it prices the plain race
    @test race_probabilities(mu; V = 0.4, D = ones(4)) ≈ want
    @test_throws ArgumentError race_probabilities(mu; V = [0.5, 0.3], D = ones(4))

    # and an ordinary rank-one loading still moves the answer
    @test !isapprox(race_probabilities(mu; V = [0.5, 0.3, -0.2, 0.1],
                                       D = ones(4)), want)
end
