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

@testset "node-rule counts" begin
    mu = [0.0, 0.3, -0.2, 0.5]
    # k = 0 is the EMPTY PRODUCT: one node of weight 1, no columns. julia
    # already fell through to it; python returned the rank-1 rule and R
    # raised, so pin it here before it drifts back
    h0 = hermite_nodes(0, 5)
    @test size(h0.F) == (1, 0)
    @test h0.W ≈ [1.0]
    @test size(hermite_nodes(1, 5).F) == (5, 1)
    # and it prices the independent race through the factor path
    @test race_probabilities(mu; V = zeros(4, 0), D = ones(4)) ≈
          race_probabilities(mu; D = ones(4))
    @test_throws ArgumentError hermite_nodes(-1, 5)
    @test_throws ArgumentError hermite_nodes(1, 0)
    @test_throws ArgumentError hermite_nodes(1, -3)
end

@testset "zero-rank loadings reach top-k" begin
    # _topk_factor_nodes special-cased rank 1 and sent every other rank
    # through the rank-2 tensor, so an (n, 0) matrix was integrated over
    # a factor space it does not have (#309)
    mu = [0.0, 0.3, -0.2, 0.5]
    D = ones(4)
    @test top_k_probabilities(mu, 2; V = zeros(4, 0), D = D) ≈
          top_k_probabilities(mu, 2; D = D)
    # and a rank-one loading still moves it
    moved = top_k_probabilities(mu, 2; V = reshape([0.9, -0.4, 0.2, -0.7], 4, 1),
                                D = D)
    @test maximum(abs.(moved .- top_k_probabilities(mu, 2; D = D))) > 0.005
end

@testset "factor-core batch (#100 #370 #444 #110)" begin
    # #100: the inverse is unit-equivariant (was log residual 51.6 here)
    p = [0.60, 0.25, 0.15]
    D = [0.005, 0.010, 0.020]
    mu = abilities_from_race(p; D = D)
    q = race_probabilities(mu; D = D)
    @test maximum(abs.(log.(q) .- log.(p))) < 1e-7
    mu100 = abilities_from_race(p; D = D .* 100^2)
    @test maximum(abs.(mu100 ./ 100 .- mu)) < 1e-6
    # #370: the top-k window is scale-free
    m = [-.5, .2, .8, -.1]; s = [.7, 1.1, .9, 1.3]
    @test maximum(abs.(top_k_probabilities(1e-18 .* m, 2; D = (1e-18 .* s) .^ 2) .-
                       top_k_probabilities(m, 2; D = s .^ 2))) < 1e-9
    # #444: a lattice needs two points
    @test_throws ArgumentError race_probabilities(m; D = ones(4), points = 1)
    @test_throws ArgumentError race_probabilities(m; D = ones(4), points = 0)
    # #110: non-finite targets are refused
    @test_throws ErrorException abilities_from_race([0.5, NaN, 0.5])
    # #317: fractional ranks are refused by name; #378: the middle rank
    @test_throws ArgumentError abilities_from_rank_marginal([.4, .3, .2, .1], 1.5)
    P = rank_probabilities([-1, -.4, .05, .45, .9]; D = ones(5), points = 257)
    @test_throws ArgumentError abilities_from_rank_marginal(P[:, 3], 3; D = ones(5),
                                                            points = 257)
end

@testset "rank marginals of a factor-correlated race (#202)" begin
    # rank_probabilities had no V/qa: the correlated call was a keyword
    # error while top_k_probabilities priced the same model
    mu = [-0.4, -0.1, 0.2, 0.5]
    D = fill(0.2, 4)
    V = reshape([-2.0, -0.5, 0.5, 2.0], 4, 1)
    R = rank_probabilities(mu; V = V, D = D, qa = 15)
    q1 = top_k_probabilities(mu, 1; V = V, D = D, qa = 15)
    q2 = top_k_probabilities(mu, 2; V = V, D = D, qa = 15)
    @test maximum(abs.(R[:, 1] .- q1)) < 1e-6
    @test maximum(abs.(vec(sum(R[:, 1:2], dims = 2)) .- q2)) < 1e-6
    # the Python reference on the same fixture
    @test maximum(abs.(R[:, 1] .- [0.53003131, 0.08960483, 0.049621,
                                   0.33074286])) < 1e-6
    @test abs(R[4, 1] - rank_probabilities(mu; D = D)[4, 1]) > 0.2
    V2 = hcat(V, [0.3, -0.2, 0.4, -0.5])
    R2 = rank_probabilities(mu; V = V2, D = D, qa = 9)
    @test maximum(abs.(R2[:, 1] .- top_k_probabilities(mu, 1; V = V2, D = D,
                                                       qa = 9))) < 1e-6
    @test rank_probabilities(mu; V = zeros(4, 0), D = D) ≈
          rank_probabilities(mu; D = D)
end

@testset "Halton bases past the old 12-prime table (#402)" begin
    @test winning._first_primes(13) ==
          [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    for r in (12, 13, 20)
        hw = winning.halton_normal_nodes(r, 64)
        @test size(hw.F) == (64, r) && all(isfinite, hw.F)
    end
    n = 14
    mu = collect(range(-0.3, 0.3; length = n))
    V = 0.15 .* [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:(n - 1)]
    p = race_probabilities(mu; V = V, D = ones(n), points = 257)
    # Python (Sobol) on the same fixture
    ref = [0.11508239, 0.10653908, 0.09851713, 0.09099268, 0.08394499,
           0.07735006, 0.07118896, 0.06543323, 0.06007212, 0.05507979,
           0.05043980, 0.04613063, 0.04213669, 0.03709245]
    @test length(p) == n && abs(sum(p) - 1) < 1e-8
    @test maximum(abs.(p .- ref)) < 2e-3
end

@testset "weights that overflow only in their sum (#415)" begin
    mu = [0.0, 0.4, 1.0]
    V = reshape([0.7, 0.0, -0.4], 3, 1)
    F = reshape([-1.0, 1.0], 2, 1)
    p1 = race_probabilities(mu; V = V, D = ones(3), F = F, W = [1.0, 1.0],
                            points = 257)
    pb = race_probabilities(mu; V = V, D = ones(3), F = F,
                            W = [1e308, 1e308], points = 257)
    @test all(isfinite, pb)
    @test maximum(abs.(pb .- p1)) < 1e-15
    @test maximum(abs.(p1 .- [0.51529828, 0.30546137, 0.17924036])) < 1e-6
    @test_throws ArgumentError race_probabilities(mu; V = V, D = ones(3),
                                                  F = F, W = [0.0, 0.0])
end
