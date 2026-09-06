# Fixture parity (python winning.likelihood/mnprobit is the spec),
# score vs finite differences AND vs ForwardDiff duals, GHK-vs-exact
# agreement, and an end-to-end fit recovery.
#   julia --project=julia/MultinomialProbit/test julia/MultinomialProbit/test/runtests.jl
using Test
using Random: Xoshiro
import LinearAlgebra
import ForwardDiff

include(joinpath(@__DIR__, "..", "src", "MultinomialProbit.jl"))
using .MultinomialProbit
const MP = MultinomialProbit

# minimal JSON reader (same recipe as parity/check.jl)
include(joinpath(@__DIR__, "minijson.jl"))

vecs = parse_json(read(joinpath(@__DIR__, "vectors.json"), String))
T, J, p, r = Int(vecs["T"]), Int(vecs["J"]), Int(vecs["p"]), Int(vecs["r"])
X = zeros(T, J, p)
for t in 1:T, j in 1:J, c in 1:p
    X[t, j, c] = vecs["X"][t][j][c]
end
choice = Int.(vecs["choice"]) .+ 1

@testset "fixture parity with the python reference" begin
    for case in vecs["cases"]
        beta = Float64.(case["beta"])
        V = permutedims(hcat([Float64.(row) for row in case["V"]]...))
        mu = zeros(T, J)
        for c in 1:p
            mu .+= X[:, :, c] .* beta[c]
        end
        ll, dmu, dV = choice_loglik_and_score(mu, V, choice)
        @test abs(ll - case["loglik"]) < 1e-9
        refmu = permutedims(hcat([Float64.(row) for row in case["dmu"]]...))
        refV = permutedims(hcat([Float64.(row) for row in case["dV"]]...))
        @test maximum(abs.(dmu .- refmu)) < 1e-9
        @test maximum(abs.(dV .- refV)) < 1e-9
        # forward choice probabilities, first five observations
        m5 = MNProbit(X[1:5, :, 1:p], choice[1:5]; intercepts = false, r = r)
        m5.beta = beta
        m5.V = V
        P = predict_proba(m5)
        refP = permutedims(hcat([Float64.(row) for row in case["proba5"]]...))
        @test maximum(abs.(P .- refP)) < 1e-9
    end
end

@testset "analytic score vs central differences" begin
    beta = [0.4, -0.2]
    V = [0.0 0.0; 0.5 0.0; -0.3 0.4; 0.1 -0.2]
    mu = zeros(T, J)
    for c in 1:p
        mu .+= X[:, :, c] .* beta[c]
    end
    _, dmu, dV = choice_loglik_and_score(mu, V, choice)
    h = 1e-6
    for (t, j) in ((1, 2), (7, 1), (23, 4))
        mp = copy(mu); mp[t, j] += h
        mm = copy(mu); mm[t, j] -= h
        fd = (choice_loglik_and_score(mp, V, choice)[1] -
              choice_loglik_and_score(mm, V, choice)[1]) / (2h)
        @test abs(fd - dmu[t, j]) < 1e-5
    end
    for (row, col) in ((2, 1), (3, 2))
        Vp = copy(V); Vp[row, col] += h
        Vm = copy(V); Vm[row, col] -= h
        fd = (choice_loglik_and_score(mu, Vp, choice)[1] -
              choice_loglik_and_score(mu, Vm, choice)[1]) / (2h)
        @test abs(fd - dV[row, col]) < 1e-5
    end
end

@testset "analytic score vs ForwardDiff (machine-precision referee)" begin
    # the finite-difference check above tunes a step and settles for
    # 1e-5; dual numbers through the SAME code settle for 1e-10
    m = MNProbit(X, choice; intercepts = true, r = r)
    for (label, theta) in (
        ("GH branch", vcat(fill(0.2, m.p), fill(0.15, length(m.pos)))),
        ("Halton branch", vcat(fill(0.1, m.p), fill(2.5, length(m.pos)))),
    )
        f(t) = begin
            beta = t[1:m.p]
            V = MP._unpack(m, t)[2]
            choice_loglik_and_score(MP._mu(m, beta), V, m.choice)[1]
        end
        g_ad = ForwardDiff.gradient(f, theta)
        nll, g_an = MP._nll_grad(m, theta)
        @test maximum(abs.(g_ad .+ g_an)) < 1e-10 * max(1.0, maximum(abs.(g_ad)))
    end
end

@testset "GHK agrees with the exact engine" begin
    beta = [0.8, -0.5]
    V = [0.0 0.0; 0.6 0.0; -0.4 0.5; 0.2 -0.3]
    m = MNProbit(X, choice; intercepts = false, r = r)
    m.beta = beta
    m.V = V
    P = predict_proba(m)
    mu = zeros(T, J)
    for c in 1:p
        mu .+= X[:, :, c] .* beta[c]
    end
    Vg = V .- sum(V, dims = 1) ./ J
    for t in (1, 5, 11)
        for k in 1:J
            pg = MP.ghk_choice_prob(vec(mu[t, :]), Vg, k;
                                    r_draws = 60000, seed = 5)
            @test abs(pg - P[t, k]) < 0.005
        end
    end
end

@testset "end-to-end fit recovers planted parameters" begin
    rng = Xoshiro(11)
    T2, J2, p2, r2 = 1500, 4, 2, 1
    X2 = randn(rng, T2, J2, p2)
    beta_true = [0.9, -0.6]
    V_true = reshape([0.0, 0.7, -0.5, 0.2], 4, 1)
    mu = zeros(T2, J2)
    for c in 1:p2
        mu .+= X2[:, :, c] .* beta_true[c]
    end
    U = mu .+ randn(rng, T2, 1) * V_true' .+ randn(rng, T2, J2)
    ch = [argmax(U[t, :]) for t in 1:T2]
    m = MNProbit(X2, ch; intercepts = false, r = r2)
    fit!(m)
    @test m.converged
    @test maximum(abs.(m.beta .- beta_true)) < 0.15
    # loadings: binary choices carry little covariance information, so
    # at T = 1500 assert the SHAPE (up to the column sign gauge: V and
    # -V price alike) and in-sample MLE dominance, not the magnitude --
    # measured: fitted loglik beats truth by ~5 nats with correlation
    # 0.96, loadings inflated ~1.5x (sampling variance, not a bug)
    v = m.V[:, 1] .- sum(m.V[:, 1]) / J2
    vt = V_true[:, 1] .- sum(V_true[:, 1]) / J2
    corr = sum(v .* vt) / sqrt(sum(abs2, v) * sum(abs2, vt))
    @test abs(corr) > 0.9
    mu_t = zeros(T2, J2)
    for c in 1:p2
        mu_t .+= X2[:, :, c] .* beta_true[c]
    end
    ll_true, _, _ = choice_loglik_and_score(mu_t, V_true, ch)
    @test m.loglik >= ll_true - 1e-6
end

println("all MultinomialProbit tests passed")

@testset "inference: vcov, stderror, per-observation scores" begin
    rng = Xoshiro(3)
    T3, J3 = 800, 4
    X3 = randn(rng, T3, J3, 2)
    beta_true = [0.7, -0.4]
    V_true = reshape([0.0, 0.5, -0.4, 0.2], 4, 1)
    mu = zeros(T3, J3)
    for c in 1:2
        mu .+= X3[:, :, c] .* beta_true[c]
    end
    U = mu .+ randn(rng, T3, 1) * permutedims(V_true) .+ randn(rng, T3, J3)
    ch = [argmax(U[t, :]) for t in 1:T3]
    m = MNProbit(X3, ch; intercepts = false, r = 1)
    fit!(m)
    # per-observation scores sum to the total analytic score
    G = score_matrix(m)
    _, g_tot = MP._nll_grad(m, m.theta)
    @test maximum(abs.(vec(sum(G, dims = 1)) .+ g_tot)) < 1e-8
    # hessian: symmetric by construction, negative definite at the MLE
    Hs = loglik_hessian(m)
    ev = maximum(real.(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(Hs))))
    @test ev < 0
    # FD-of-score hessian vs the ForwardDiff dual engine
    Hfd = copy(Hs)
    MP.HESSIAN_ENGINE[] = (mm, tt) -> begin
        g(t) = -MP._nll_grad(mm, t)[2]
        H2 = ForwardDiff.jacobian(g, tt)
        (H2 .+ H2') ./ 2
    end
    Had = loglik_hessian(m)
    MP.HESSIAN_ENGINE[] = nothing
    @test maximum(abs.(Hfd .- Had)) < 1e-5 * max(1.0, maximum(abs.(Had)))
    # the three vcov flavors agree in scale, and truth is covered
    for method in (:hessian, :opg, :sandwich)
        se = stderror(m; method = method)
        @test all(isfinite, se) && all(se .> 0)
        @test maximum(abs.(m.beta .- beta_true) ./ se[1:2]) < 4.0
    end
    # show() smoke
    io = IOBuffer()
    show(io, MIME"text/plain"(), m)
    @test occursin("beta[1]", String(take!(io)))
end
