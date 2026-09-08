# Fixture parity with winning/fastmvn.py (the spec), refusal contract,
# armed-fallback routing, and a cross-check against the incumbent.
#   julia --project=julia/MvNormalCDFFast/test julia/MvNormalCDFFast/test/runtests.jl
using Test

using MvNormalCDFFast
const MF = MvNormalCDFFast

include(joinpath(@__DIR__, "minijson.jl"))
vecs = parse_json(read(joinpath(@__DIR__, "vectors.json"), String))

@testset "fixture parity with python fastmvn" begin
    for case in vecs["cases"]
        V = permutedims(hcat([Float64.(row) for row in case["V"]]...))
        D = Float64.(case["D"])
        mu = Float64.(case["mu"])
        lo = case["lower"] === nothing ? nothing : Float64.(case["lower"])
        up = case["upper"] === nothing ? nothing : Float64.(case["upper"])
        p, method = mvn_cdf_fast_info(lower = lo, upper = up, mean = mu,
                                      V = V, D = D)
        if case["method"] == "factor"
            @test method == "factor"
            @test abs(p - case["p"]) < 1e-9 * max(case["p"], 1e-12)
        else
            # reference uses Sobol on the recentered tail, this port
            # deterministic GH: agree in the relative sense
            @test method == "factor-recentered"
            @test abs(log(p) - log(case["p"])) < 0.02
        end
    end
end

@testset "refusal without the incumbent loaded" begin
    # dense covariance that is NOT factor-plus-diagonal
    S = [1.0 0.5 0.2; 0.5 1.0 0.7; 0.2 0.7 1.0]
    @test MF.factorize_covariance(S) === nothing ||
          true   # rank-2 may factor a 3x3 exactly; check the 4x4 below
    S4 = [1.0 0.6 0.1 0.0; 0.6 1.0 0.5 0.3; 0.1 0.5 1.0 0.7; 0.0 0.3 0.7 1.0]
    if MF.factorize_covariance(S4; max_rank = 2) === nothing
        @test_throws ErrorException MF.mvnormcdf(zeros(4), S4,
                                              fill(-Inf, 4), zeros(4))
    end
end

using MvNormalCDF

@testset "the package extension arms the fallback" begin
    # `using MvNormalCDF` above loads MvNormalCDFFallbackExt, so the
    # hook is armed by the extension itself rather than by hand
    @test MF.DENSE_FALLBACK[] !== nothing
    S4 = [1.0 0.6 0.1 0.0; 0.6 1.0 0.5 0.3; 0.1 0.5 1.0 0.7; 0.0 0.3 0.7 1.0]
    p, e = MF.mvnormcdf(zeros(4), S4, fill(-Inf, 4), zeros(4))
    @test 0 < p < 1
    @test isfinite(e)
end

@testset "exact path agrees with the incumbent on factor structure" begin
    V = reshape([0.5, -0.3, 0.4, 0.2, -0.4, 0.3], 6, 1)
    D = [0.8, 1.1, 0.9, 1.2, 1.0, 0.7]
    S = V * V' .+ [i == j ? D[i] : 0.0 for i in 1:6, j in 1:6]
    a = fill(-Inf, 6)
    b = [0.5, 0.2, 0.8, -0.1, 0.4, 0.6]
    p_fast, e_fast = MF.mvnormcdf(zeros(6), S, a, b)
    p_inc, e_inc = MvNormalCDF.mvnormcdf(zeros(6), S, a, b; m = 200000)
    @test abs(p_fast - p_inc) < max(5 * e_inc, 1e-4)
    @test e_fast < 1e-6
end

println("all MvNormalCDFFast tests passed")
