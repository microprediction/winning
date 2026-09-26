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


# --- empty and degenerate rectangles (#235) ---------------------------
#
# P(lower <= X <= upper) over a reversed coordinate is an EMPTY event.
# Every port formed the negative conditional cell Phi(0) - Phi(1) and
# clamped it to the underflow floor, so an impossible observation came
# back as 1e-300 -- a finite log probability near -690.8 rather than
# -Inf. mvtnorm::pmvnorm, which the R port is a drop-in for, raises on
# reversed bounds and returns 0 when a coordinate has lower == upper.
@testset "empty and degenerate rectangles" begin
    @test_throws ArgumentError FactorMvNormalCDF.mvn_cdf_fast(
        lower = [1.0], upper = [0.0], mean = [0.0],
        V = zeros(1, 1), D = ones(1))

    err = try
        FactorMvNormalCDF.mvn_cdf_fast(
            lower = [0.0, 1.0, -1.0], upper = [1.0, 0.0, 1.0],
            mean = zeros(3), V = zeros(3, 1), D = ones(3))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("coordinate 2", err)

    p, meth = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = [0.5], upper = [0.5], mean = [0.0],
        V = zeros(1, 1), D = ones(1))
    @test p === 0.0                       # exactly, not 1e-300
    @test meth == "degenerate-rectangle"
    @test log(p) == -Inf

    p3, _ = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = [0.0, 0.3, -1.0], upper = [1.0, 0.3, 1.0],
        mean = zeros(3), V = zeros(3, 1), D = ones(3))
    @test p3 === 0.0

    # a valid rectangle is untouched: independent coordinates, exact
    D = [0.7, 0.9, 1.1]
    lo = [-1.0, -1.0, -1.0]; up = [0.4, 0.8, 1.2]
    pv, _ = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = lo, upper = up, mean = zeros(3),
        V = zeros(3, 1), D = D)
    # Distributions is not a test dependency; use the package's own ndtr
    Phi = FactorMvNormalCDF.ndtr
    exact = prod(Phi(up[i] / sqrt(D[i])) - Phi(lo[i] / sqrt(D[i]))
                 for i in 1:3)
    @test abs(pv - exact) < 1e-10
end

# --- scalar bounds broadcast, as in the specification (#184) ----------
#
# winning.fastmvn broadcasts a scalar bound to every coordinate, and the
# R port does the same with rep_len. This port called collect on the
# argument, and a scalar Float64 is not iterable, so the ordinary
# joint-CDF spelling upper=0.0 threw before evaluating anything.
@testset "scalar bounds broadcast" begin
    V = zeros(3, 1); D = ones(3)
    @test FactorMvNormalCDF.mvn_cdf_fast(V = V, D = D, upper = 0.0) ≈ 0.125 atol = 1e-10
    @test FactorMvNormalCDF.mvn_cdf_fast(V = V, D = D, upper = [0.0]) ≈ 0.125 atol = 1e-10
    @test FactorMvNormalCDF.mvn_cdf_fast(V = V, D = D, upper = zeros(3)) ≈ 0.125 atol = 1e-10
    # a scalar lower bound broadcasts too, and the pair agree
    a = FactorMvNormalCDF.mvn_cdf_fast(V = zeros(2, 1), D = ones(2),
                                       lower = -1.0, upper = 1.0)
    b = FactorMvNormalCDF.mvn_cdf_fast(V = zeros(2, 1), D = ones(2),
                                       lower = [-1.0, -1.0], upper = [1.0, 1.0])
    @test a ≈ b atol = 1e-12
    # a length that is neither 1 nor n is a mistake, not a recycling rule
    @test_throws ArgumentError FactorMvNormalCDF.mvn_cdf_fast(
        V = V, D = D, upper = [0.0, 1.0])
end

# --- deterministic coordinates (#206) ---------------------------------
#
# A coordinate with zero idiosyncratic variance is DETERMINISTIC given
# the factor draw, so its cell is an indicator, not a gaussian interval.
# Dividing by sd = 0 is 0/0 = NaN the moment that value lands exactly on
# an inclusive boundary: P(X_1 <= 0) = 1 for X_1 identically 0.
@testset "deterministic coordinates" begin
    fixed = (mean = [0.0, 0.0], V = zeros(2, 1), D = [0.0, 1.0])
    p, _ = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = [-Inf, -Inf], upper = [0.0, 0.0]; fixed...)
    @test !isnan(p)
    @test p ≈ 0.5 atol = 1e-10

    pin, _ = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = [-Inf, -Inf], upper = [1.0, 0.0]; fixed...)
    @test pin ≈ 0.5 atol = 1e-10

    for (lo, up) in (([-Inf, -Inf], [-1.0, 0.0]), ([0.5, -Inf], [Inf, 0.0]))
        pz, meth = FactorMvNormalCDF.mvn_cdf_fast_info(
            lower = lo, upper = up; fixed...)
        @test pz === 0.0          # not the TINY the cell floor would give
        @test meth == "outside-support"
    end

    pall, _ = FactorMvNormalCDF.mvn_cdf_fast_info(
        lower = [-Inf, -Inf], upper = [0.0, 0.0],
        mean = [0.0, 0.0], V = zeros(2, 1), D = [0.0, 0.0])
    @test pall ≈ 1.0 atol = 1e-10
end

@testset "rank zero is the independent rectangle" begin
    # every r != 1 was built as RANK TWO, so r = 0 made a spurious
    # Q^2 x 2 node matrix and then failed multiplying it by a 0 x n
    # loading transpose. The exact answer is the independent product
    # (#68).
    F, W = FactorMvNormalCDF._gh_nodes(0, 15)
    @test size(F) == (1, 0)
    @test W ≈ [1.0]
    for n in (2, 3, 4)
        @test mvn_cdf_fast(upper = zeros(n), V = zeros(n, 0),
                           D = ones(n)) ≈ 0.5 ^ n atol = 1e-10
    end
    @test_throws ArgumentError FactorMvNormalCDF._gh_nodes(-1, 15)
end

println("all FactorMvNormalCDF tests passed")
