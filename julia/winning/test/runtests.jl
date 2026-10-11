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

@testset "an overflowing target sum is rescued (#326)" begin
    # finite entries whose SUM overflows describe the same law as their
    # ratios; python rescales by the max first (#300) and so does julia
    a = abilities_from_race([4e307, 2e307, 1e307, 1e307])
    b = abilities_from_race([4.0, 2.0, 1.0, 1.0])
    @test all(isfinite, a)
    @test a ≈ b
    @test maximum(abs.(abilities_from_race(fill(1e308, 4)))) < 1e-8
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

@testset "adaptive node schedule matches GH_RULE (#528)" begin
    # the 8-runner stress field at centred sharpness 21.07, ceil(8s) = 169:
    # julia kept Gauss-Hermite to Q = 201 and moved the leaders by 42bp
    mu = [-1.2778325156570909, -1.0099930645736255, -0.2838848030639649,
          -0.22632464605522293, -0.11596618882209474, -0.10779858154488295,
          0.20904942336288942, 1.0204595606925912]
    v = [-0.8571701970944656, 3.3310423958253588, 0.2338294924083979,
         -0.34458791516111925, -0.27324453897087425, -0.6600034669284740,
         -1.0471076720246453, -0.3827580980541786]
    V = reshape(v, :, 1)
    D = fill(0.05, 8)
    p = race_probabilities(mu; V = V, D = D, points = 257)
    Q = 169
    Fmid = reshape(winning.qnorm.(((1:Q) .- 0.5) ./ Q), :, 1)
    pmid = race_probabilities(mu; V = V, D = D, F = Fmid, W = fill(1.0 / Q, Q),
                              points = 257)
    @test maximum(abs.(p .- pmid)) < 1e-14          # the midpoint family
    mc = [0.5254543333, 0.4744646667]               # 3M-path reference
    @test maximum(abs.(p[1:2] .- mc)) < 3e-4        # GH-169 was 4.2e-3 off
    # rank 2 between the old 3.0 and the reference 3.75 cutover stays
    # Gauss-Hermite at the capped order
    mu3 = [0.0, 0.2, -0.1]
    V2 = [1.0 0.5; -0.6 0.4; -0.4 -0.9]
    Vc = V2 .- sum(V2, dims = 1) ./ 3
    s = sqrt(2) * maximum(sqrt.(vec(sum(Vc .^ 2, dims = 2))))
    D2 = fill((s / 3.4)^2, 3)                        # sharpness 3.4
    q = Int(min(max(ceil(8 * 3.4), 15), 41))
    h = hermite_nodes(2, q)
    @test race_probabilities(mu3; V = V2, D = D2, points = 257) ≈
          race_probabilities(mu3; V = V2, D = D2, F = h.F, W = h.W,
                             points = 257) atol = 1e-14
end

@testset "scalar D through every top-k/rank entry point (#485)" begin
    mu = [0.0, 0.3, -0.2, 0.5]
    Jv = top_k_jacobians(mu, 2; D = ones(4), points = 257)
    Js = top_k_jacobians(mu, 2; D = 1.0, points = 257)
    @test Js.Jmu == Jv.Jmu && Js.Jsigma == Jv.Jsigma
    V = reshape([0.4, -0.2, 0.1, -0.3], 4, 1)
    @test top_k_jacobians(mu, 2; D = 1.0, V = V, points = 257).Jmu ==
          top_k_jacobians(mu, 2; D = ones(4), V = V, points = 257).Jmu
    @test top_k_probabilities(mu, 2; D = 2.0, points = 257) ==
          top_k_probabilities(mu, 2; D = fill(2.0, 4), points = 257)
    @test rank_probabilities(mu; D = 1.0, points = 257) ==
          rank_probabilities(mu; D = ones(4), points = 257)
    q1 = top_k_probabilities(mu, 1; points = 257)
    q2 = top_k_probabilities(mu, 2; points = 257)
    a = loc_scale_from_topk_pair(q1, 1, q2, 2; D0 = 1.0, points = 257)
    b = loc_scale_from_topk_pair(q1, 1, q2, 2; D0 = ones(4), points = 257)
    @test a.mu == b.mu && a.sd == b.sd
    for bad in ([1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 1.0],
                [1.0, NaN, 1.0, 1.0], [1.0, Inf, 1.0, 1.0], 0.0)
        @test_throws ArgumentError top_k_jacobians(mu, 2; D = bad, points = 257)
        @test_throws ArgumentError top_k_probabilities(mu, 2; D = bad,
                                                       points = 257)
        @test_throws ArgumentError loc_scale_from_topk_pair(q1, 1, q2, 2;
                                                            D0 = bad,
                                                            points = 257)
    end
end

@testset "exact loc/scale warm start returns in the gauge (#595)" begin
    mu = [-3.0, -1.0, 1.0, 3.0]
    sd = [2.0, 3.0, 4.0, 5.0]
    D = sd .^ 2
    q1 = top_k_probabilities(mu, 1; D = D, points = 1025)
    q2 = top_k_probabilities(mu, 2; D = D, points = 1025)
    fit = loc_scale_from_topk_pair(q1, 1, q2, 2; D0 = D, mu0 = mu,
                                   points = 1025, return_info = true)
    @test fit.converged
    @test abs(sum(fit.mu)) < 1e-12
    @test abs(sum(log.(fit.sd))) < 1e-12
    c = exp(sum(log.(sd)) / 4)
    @test fit.mu ≈ mu ./ c atol = 1e-12
    @test fit.sd ≈ sd ./ c atol = 1e-12
    @test fit.sd ≈ [0.604275079471354, 0.906412619207030,
                    1.208550158942707, 1.510687698678384] atol = 1e-12
    p2 = q2 .- q1
    w = loc_scale_from_win_and_second(q1, p2; D0 = D, mu0 = mu,
                                      points = 1025, return_info = true)
    @test abs(sum(w.mu)) < 1e-12 && abs(sum(log.(w.sd))) < 1e-12
end
