# Fixture parity with winning/fastmvn.py (the spec), delegation of
# every case outside the exact path to MvNormalCDF, and a cross-check
# against MvNormalCDF on factor structure.
#   julia -e 'using Pkg; Pkg.develop(path = "julia/FactorMvNormalCDF"); Pkg.test("FactorMvNormalCDF")'
using Test

using FactorMvNormalCDF
import MvNormalCDF
const MF = FactorMvNormalCDF

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

@testset "own name only: mvnormcdf is not exported" begin
    @test :mvnormcdf_factor in names(FactorMvNormalCDF)
    @test !(:mvnormcdf in names(FactorMvNormalCDF))
end

@testset "cases outside the exact path are delegated, not refused" begin
    # a dense covariance that is not factor-plus-diagonal at rank <= 2
    S4 = [1.0 0.6 0.1 0.0; 0.6 1.0 0.5 0.3; 0.1 0.5 1.0 0.7; 0.0 0.3 0.7 1.0]
    p, e = mvnormcdf_factor(zeros(4), S4, fill(-Inf, 4), zeros(4))
    @test 0 < p < 1
    @test isfinite(e)
    p2, method = mvn_cdf_fast_info(sigma = S4, upper = zeros(4))
    @test method == "fallback"
    @test 0 < p2 < 1
    # keyword arguments reach MvNormalCDF: a tiny QMC budget still runs
    p3, e3 = mvnormcdf_factor(zeros(4), S4, fill(-Inf, 4), zeros(4); m = 2000)
    @test 0 < p3 < 1
    # rank > 2 is delegated as well
    V3 = [0.4 0.1 0.2; -0.3 0.2 0.1; 0.2 -0.4 0.3; 0.1 0.3 -0.2; -0.2 0.1 0.4]
    D3 = fill(1.0, 5)
    p4, method4 = mvn_cdf_fast_info(V = V3, D = D3, upper = zeros(5))
    @test method4 == "fallback"
    @test 0 < p4 < 1
end

@testset "exact path agrees with MvNormalCDF on factor structure" begin
    V = reshape([0.5, -0.3, 0.4, 0.2, -0.4, 0.3], 6, 1)
    D = [0.8, 1.1, 0.9, 1.2, 1.0, 0.7]
    S = V * V' .+ [i == j ? D[i] : 0.0 for i in 1:6, j in 1:6]
    a = fill(-Inf, 6)
    b = [0.5, 0.2, 0.8, -0.1, 0.4, 0.6]
    p_fast, e_fast = mvnormcdf_factor(zeros(6), S, a, b)
    p_inc, e_inc = MvNormalCDF.mvnormcdf(zeros(6), S, a, b; m = 200000)
    @test abs(p_fast - p_inc) < max(5 * e_inc, 1e-4)
    @test e_fast < 1e-6
end

println("all FactorMvNormalCDF tests passed")
