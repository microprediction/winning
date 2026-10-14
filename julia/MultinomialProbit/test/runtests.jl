# Fixture parity (python winning.likelihood/mnprobit is the spec),
# score vs finite differences AND vs ForwardDiff duals, GHK-vs-exact
# agreement, and an end-to-end fit recovery.
#   julia --project=julia/MultinomialProbit/test julia/MultinomialProbit/test/runtests.jl
using Test
using Random: Xoshiro, randperm
import LinearAlgebra
import ForwardDiff

using MultinomialProbit
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

# --- every observation is accounted for (#194) -------------------------
#
# The core iterates over the LEGAL labels and gathers the rows matching
# each, so a row whose choice is outside 1..J is never visited: it
# contributed nothing to the log-likelihood and a zero row to the score,
# and the fit silently optimised a SUBSET while reporting it as the
# whole. This port also accepted a choice vector SHORTER than T and
# dropped the tail. Dropping observations RAISES the log-likelihood,
# because there is less of it, so nothing downstream can tell it from a
# better fit.
@testset "choice validation" begin
    T, J, r = 40, 3, 1
    rng = Xoshiro(1)
    mu = randn(rng, T, J)
    V = randn(rng, J, r) .* 0.4
    good = rand(rng, 1:J, T)

    ll, _, _ = MultinomialProbit.choice_loglik_and_score(mu, V, good)
    @test isfinite(ll)

    # fewer observations is a BETTER number -- the reason silence hurts
    part, _, _ = MultinomialProbit.choice_loglik_and_score(
        mu[6:end, :], V, good[6:end])
    @test part > ll

    for bad in (J + 1, 99, 0, -1)
        ch = copy(good)
        ch[1] = bad
        @test_throws ArgumentError MultinomialProbit.choice_loglik_and_score(
            mu, V, ch)
    end

    # a SHORT vector was silently truncated here, unlike the other ports
    @test_throws ArgumentError MultinomialProbit.choice_loglik_and_score(
        mu, V, good[1:30])
    @test_throws ArgumentError MultinomialProbit.choice_loglik_and_score(
        mu, V, vcat(good, 1))

    err = try
        ch = copy(good); ch[4] = 99
        MultinomialProbit.choice_loglik_and_score(mu, V, ch)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("choice[4]", err)
end

# --- new data must carry the fitted design (#195) ----------------------
#
# predict_proba decided whether to prepend generated intercept columns
# from the COLUMN COUNT alone, and the model recorded neither whether it
# had generated them nor how many covariates the caller supplied. New
# data with the wrong number of features were therefore silently
# REINTERPRETED: a model fitted on two covariates without intercepts,
# handed one covariate, prepended two synthetic intercept columns and
# then used only the first p of the three -- applying coefficients
# fitted to the covariates to the intercepts, ignoring the supplied
# feature, and returning a plausible probability row.
@testset "predict_proba checks the design" begin
    X = zeros(2, 3, 2)
    m = MNProbit(X, [1, 2]; intercepts = false, r = 1)
    m.beta .= [0.7, -0.4]
    @test m.p_raw == 2
    @test m.intercepts == false

    # one covariate where two were fitted: used to return a probability row
    @test_throws DimensionMismatch predict_proba(
        m; X = reshape([1.0, 2.0, 3.0], 1, 3, 1))
    # three where two were fitted
    @test_throws DimensionMismatch predict_proba(m; X = randn(1, 3, 3))
    # the wrong number of ALTERNATIVES
    @test_throws DimensionMismatch predict_proba(m; X = randn(1, 2, 2))

    # the right design still works, and is a distribution
    P = predict_proba(m; X = reshape(collect(1.0:6.0), 1, 3, 2))
    @test size(P) == (1, 3)
    @test all(P .>= 0)
    @test abs(sum(P) - 1) < 1e-10

    # a model WITH generated intercepts takes either spelling
    Xi = randn(6, 3, 2)
    mi = MNProbit(Xi, [1, 2, 3, 1, 2, 3]; intercepts = true, r = 1)
    @test mi.intercepts && mi.p_raw == 2 && mi.p == 2 + (3 - 1)
    for nc in (mi.p_raw, mi.p)
        Q = predict_proba(mi; X = randn(4, 3, nc))
        @test size(Q) == (4, 3)
        @test maximum(abs.(sum(Q; dims = 2) .- 1)) < 1e-10
    end
    # and nothing in between
    @test_throws DimensionMismatch predict_proba(mi; X = randn(4, 3, 3))
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

# --- the tail likelihood is finite and its score is not zero (#270) ---
#
# lp was log.(max.(ndtr.(Aj), TINY)). ndtr underflows to exactly zero
# below about -37, so every deep tail collapsed to the SAME -690.78 and
# the objective went FLAT; the Mills ratio then underflowed to a score
# of exactly ZERO, so an optimizer declares convergence precisely where
# an observation is most badly contradicted.
#
# logndtr removes both. The quadrature error underneath (the fixed rule
# sampled the own noise where the prior has mass) is closed by the
# closed-form binary and the tilted own-noise nodes.
@testset "tail likelihood is finite with a nonzero score" begin
    binary(gap) = choice_loglik_and_score(
        reshape([gap, 0.0], 1, 2), zeros(2, 1), [1])

    lls = [binary(g)[1] for g in (-40.0, -60.0, -100.0, -200.0)]
    @test all(isfinite, lls)
    @test all(diff(lls) .< -1.0)          # strictly falling, by a lot
    @test minimum(lls) < -1e4             # far past the old -690.78

    prev = 0.0
    for gap in (-40.0, -60.0, -100.0, -200.0)
        _, dmu, _ = binary(gap)
        @test all(isfinite, dmu)
        @test dmu[1, 1] > 0.0             # not the zero score
        @test dmu[1, 1] > prev            # grows with the surprise
        prev = dmu[1, 1]
    end

    # where the rule does reach the integrand the answer is right, so
    # the tail failure is the quadrature and not the formula
    for gap in (-1.0, -2.0, -3.0)
        ll, _, _ = binary(gap)
        ex = MP.logndtr(gap / sqrt(2))
        @test abs(ll - ex) / abs(ex) < 1e-3
    end

    # was 38% away at a gap of -20: J = 2 is now the closed form, and
    # J >= 3 tilts the own-noise nodes to the integrand's mode
    ll20, _, _ = binary(-20.0)
    ex20 = MP.logndtr(-20.0 / sqrt(2))
    @test abs(ll20 - ex20) <= 1e-12 * abs(ex20)
    for gap in (-3.0, -8.0, -20.0, -60.0)
        llt, dmt, _ = choice_loglik_and_score(
            reshape([gap, 0.0, -80.0], 1, 3), zeros(3, 1), [1])
        ext = MP.logndtr(gap / sqrt(2))
        @test abs(llt - ext) <= 1e-7 * abs(ext)
        es = exp(-0.5 * (gap / sqrt(2))^2 - 0.5 * log(2pi) - ext) / sqrt(2)
        @test abs(dmt[1, 1] - es) <= 1e-5 * es
    end

    # logndtr itself, against the naive form where the naive form works
    for z in (-3.0, -1.0, 0.0, 0.5, 2.0)
        @test isapprox(MP.logndtr(z), log(MP.ndtr(z)); rtol = 1e-12)
    end
    @test MP.logndtr(-40.0) < -700.0      # past where ndtr underflows
    @test isfinite(MP.logndtr(-300.0))
end

# --- GHK: homogeneous in the utility unit, D is a variance ------------
# Binary races have one truncation step, so these are exact checks.
@testset "GHK is scale free and blind to a common shock (#409, #302)" begin
    exact = MP.ndtr(1 / sqrt(2))
    for c in (1.0, 1e-6, 1e-8)
        # the absolute 1e-12 ridge gave 0.7181 at 1e-6 and 0.5040 at 1e-8
        p = MP.ghk_choice_prob([0.0, c], zeros(2, 0), 2;
                               D = [c^2, c^2], r_draws = 32, seed = 9)
        @test abs(p - exact) < 1e-12
    end
    for v in (1e4, 1e8)
        p = MP.ghk_choice_prob([0.0, 1.0], fill(v, 2, 1), 2;
                               D = [1.0, 1.0], r_draws = 8, seed = 9)
        @test abs(p - exact) < 1e-12
    end
end

@testset "GHK keeps a valid small pivot and subnormal scales (#409)" begin
    exact = MP.ndtr(1 / sqrt(2))
    for c in (1e-150, 1e-155)
        # the 1e-300 floor was an absolute 1e-312 ridge below it
        p = MP.ghk_choice_prob([0.0, c], zeros(2, 0), 2;
                               D = [c^2, c^2], r_draws = 32, seed = 9)
        @test abs(p - exact) < 1e-12
    end
    # near-twins sharing a unit factor with variance 2^-43; the always-on
    # relative ridge priced [0.5, 0.167, 0.333]
    delta = 2.0^-43
    a = 0.5 * sqrt(2 * delta)
    mu = [0.0, -a, a]
    p = [MP.ghk_choice_prob(mu, reshape([0.0, 1.0, 1.0], 3, 1), k;
                            D = [1.0, delta, delta], r_draws = 200_000,
                            seed = 7 + k) for k in 1:3]
    @test maximum(abs.(p ./ sum(p) .- [0.49999992, 0.07932763, 0.42067245])) < 3e-3
end

@testset "GHK refuses a D that is not a variance (#367)" begin
    V = zeros(2, 0)
    @test_throws ArgumentError MP.ghk_choice_prob([0.0, 1.0], V, 2;
                                                  D = [1.0, -0.5])
    @test_throws ArgumentError MP.ghk_choice_prob([0.0, 1.0], V, 2;
                                                  D = [1.0, NaN])
    @test_throws ArgumentError MP.ghk_choice_prob([0.0, 1.0], V, 2;
                                                  D = [1.0, Inf])
    # a wrong length is the DimensionMismatch of #439's shared check
    @test_throws DimensionMismatch MP.ghk_choice_prob([0.0, 1.0], V, 2;
                                                      D = [1.0])
    @test_throws DimensionMismatch MP.ghk_choice_prob([0.0, 1.0], V, 2;
                                                      D = [1.0, 1.0, 1.0])
    # round-off below zero is zero, and zero is a legal variance
    p = MP.ghk_choice_prob([0.0, 1.0], V, 2; D = [1.0, -1e-15])
    @test abs(p - MP.ndtr(1.0)) < 1e-12
end

# --- integral float labels are accepted, fractional ones refused (#194) --
@testset "choice labels: integral numerics, constructor length" begin
    mu = zeros(3, 3)
    V = zeros(3, 1)
    ll_i, _, _ = choice_loglik_and_score(mu, V, [1, 2, 3])
    ll_f, _, _ = choice_loglik_and_score(mu, V, [1.0, 2.0, 3.0])
    @test ll_i == ll_f                    # 1.0 is the label 1, as in Python/R
    for bad in ([1.5, 2.0, 3.0], [1.0, NaN, 3.0], [1.0, Inf, 3.0],
                Any[1, missing, 3], [1, 0, 4])
        @test_throws ArgumentError choice_loglik_and_score(mu, V, bad)
    end
    # the constructor recorded T = 3 while holding one choice
    @test_throws ArgumentError MNProbit(zeros(3, 3, 1), [1];
                                        intercepts = false, r = 1)
    @test_throws ArgumentError MNProbit(zeros(3, 3, 1), [1, 2, 4];
                                        intercepts = false, r = 1)
    m = MNProbit(zeros(3, 3, 1), [1.0, 2.0, 3.0]; intercepts = false, r = 1)
    @test m.choice == [1, 2, 3]
end

# --- D has exactly one entry per alternative (#439) --------------------
@testset "D length is checked before sharpness dispatch" begin
    mu = reshape([2.29264426, 0.64388188, -0.17180493, 3.74907237], 1, :)
    V = [-0.0107529507 1.80970562; -0.6565907660 0.00168241106;
         0.3133862710 0.0763054062; 0.3539574460 -1.88769344]
    ll, _, _ = choice_loglik_and_score(mu, V, [4]; D = ones(4))
    @test isfinite(ll)
    # an unused fifth entry used to switch GH -> Halton: ll moved by 0.027
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu, V, [4]; D = [1.0, 1.0, 1.0, 1.0, 1e-12])
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu, V, [4]; D = ones(3))
    @test_throws DimensionMismatch MP.ghk_choice_prob(
        vec(mu), V, 4; D = ones(5))
end

# --- OPG never touches the Hessian (#434) ------------------------------
@testset "vcov(:opg) is Hessian-free" begin
    Xo = reshape([0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0], 4, 2, 1)
    m = MNProbit(Xo, [2, 1, 2, 1]; intercepts = false, r = 0)
    m.theta .= 0.3
    m.converged = true          # a hand-set theta; inference needs it (#493)
    G = score_matrix(m)
    old = MP.HESSIAN_ENGINE[]
    try
        MP.HESSIAN_ENGINE[] = (mm, tt) -> error("HESSIAN CALLED")
        @test vcov(m; method = :opg) ≈ inv(G' * G)
        @test_throws ErrorException vcov(m; method = :hessian)
    finally
        MP.HESSIAN_ENGINE[] = old
    end
    @test_throws ErrorException vcov(m; method = :bogus)
end

# --- GHK fits get no exact-likelihood inference (#214) -----------------
@testset "inference refuses a GHK fit" begin
    rng = Xoshiro(1)
    Xg = randn(rng, 40, 3, 1)
    chg = repeat([1, 2, 3, 1], 10)
    m = MNProbit(Xg, chg; intercepts = true, r = 1)
    fit!(m; method = :ghk, r_draws = 8, seed = 9, maxiter = 5)
    @test m.method == :ghk
    @test isfinite(loglikelihood(m))
    @test_throws ArgumentError score_matrix(m)
    @test_throws ArgumentError loglik_hessian(m)
    for method in (:hessian, :opg, :sandwich)
        @test_throws ArgumentError vcov(m; method = method)
        @test_throws ArgumentError stderror(m; method = method)
    end
    io = IOBuffer()
    show(io, MIME"text/plain"(), m)
    out = String(take!(io))
    @test occursin("not available", out)
    # refitting exactly lifts the GHK refusal. This random-choice design
    # has no exact maximum (BFGS stops with -H eigenvalue -1.4e4), so the
    # #493 guards still refuse the Hessian forms; OPG under the explicit
    # override shows the #214 refusal itself is gone.
    fit!(m)
    @test m.method == :exact
    @test all(isfinite, vcov(m; method = :opg, allow_unconverged = true))
end

# --- Halton has as many bases as the integral has dimensions (#396) ----
@testset "sharp likelihood at rank 8 and beyond" begin
    @test MP._first_primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for (J, r) in ((8, 7), (9, 8), (12, 11))
        mu = zeros(1, J)
        V = zeros(J, r)
        V[J, r] = 3.0                  # sharpness > 3 -> Halton, r + 1 dims
        ll, dmu, dV = choice_loglik_and_score(mu, V, [J])
        @test isfinite(ll) && all(isfinite, dmu) && all(isfinite, dV)
    end
    # (the Gauss-Hermite side at these ranks is a 7^(r+1) tensor --
    # 40M nodes at r = 8 -- in this port and the Python reference alike;
    # it is not exercised here)
    # Python (scipy Sobol) gives probability 0.32892 for J=9, r=8, v=3
    mu = zeros(1, 9); V = zeros(9, 8); V[9, 8] = 3.0
    @test abs(exp(choice_loglik_and_score(mu, V, [9])[1]) - 0.3289249) < 0.01
end

# --- ranks at or above J - 1 are not identified (#201) -----------------
@testset "factor rank identification" begin
    # the exact ridge: contrast covariance of Vp is 4x that of V, so
    # (2 beta, Vp) prices every choice like (beta, V)
    V = [0.0 0.0; 0.5 0.0; 0.2 0.7]
    Vp = [0.0 0.0; 2.6457513110645907 0.0; 1.2850792082313727 2.543338638202044]
    A = [-1.0 1.0 0.0; -1.0 0.0 1.0]
    C(V) = A * (V * V' + LinearAlgebra.I) * A'
    @test C(Vp) ≈ 4 .* C(V)
    # GHK on the differenced covariance sees exactly the scaled law:
    # same CRN draws, Cholesky factor doubled, so equal to rounding
    mu = [0.3, -0.2, 0.1]
    for k in 1:3
        p1 = MP.ghk_choice_prob(mu, V, k; r_draws = 2000, seed = 3)
        p2 = MP.ghk_choice_prob(2 .* mu, Vp, k; r_draws = 2000, seed = 3)
        @test abs(p1 - p2) < 1e-6
    end

    @test max_identified_rank.((2, 3, 4, 5)) == (0, 1, 2, 3)
    for J in 2:5
        X = randn(Xoshiro(J), 6, J, 1)
        ch = [mod1(t, J) for t in 1:6]
        m = MNProbit(X, ch)                 # default rank is identified
        @test m.r == min(2, J - 2)
        for r in 0:(J - 2)
            @test MNProbit(X, ch; r = r).r == r
        end
        for r in (J - 1, J)
            @test_throws ArgumentError MNProbit(X, ch; r = r)
        end
    end
    # the binary case fits with r = 0
    rng = Xoshiro(4)
    Xb = randn(rng, 300, 2, 1)
    U = 0.8 .* Xb[:, :, 1] .+ randn(rng, 300, 2)
    chb = [argmax(U[t, :]) for t in 1:300]
    mb = fit!(MNProbit(Xb, chb; intercepts = false))
    @test mb.converged && abs(mb.beta[1] - 0.8) < 0.25
    @test all(isfinite, stderror(mb))
end

@testset "estimator boundary contract (#479 #495 #496 #560 #581 #591)" begin
    X = zeros(2, 4, 1)
    for bad in (1.1, 1.9, NaN, Inf, true)
        @test_throws ArgumentError MNProbit(X, [1, 2]; r = bad)
    end
    @test MNProbit(X, [1, 2]; r = 2.0).r == 2
    # zero observations
    for ic in (true, false)
        @test_throws ArgumentError MNProbit(Array{Float64}(undef, 0, 3, 0),
                                            Int[]; intercepts = ic, r = 0)
    end
    # parameter-free null model
    m0 = fit!(MNProbit(Array{Float64}(undef, 4, 2, 0), [1, 2, 1, 2];
                       intercepts = false, r = 0))
    @test m0.converged && isempty(coef(m0))
    @test isapprox(m0.loglik, -4 * log(2); atol = 1e-12)
    # maxiter / r_draws are whole counts
    d = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    Xo = reshape(hcat(-d ./ 2, d ./ 2), 6, 2, 1)
    cho = [1, 1, 2, 1, 2, 2]
    for bad in (NaN, Inf, 0.5, -1, true)
        @test_throws ArgumentError fit!(MNProbit(Xo, cho; intercepts = false,
                                                 r = 0); maxiter = bad)
    end
    @test_throws ArgumentError fit!(MNProbit(Xo, cho; intercepts = false,
                                             r = 0); method = :ghk, r_draws = 0)
    mz = fit!(MNProbit(Xo, cho; intercepts = false, r = 0); maxiter = 0)
    @test !mz.converged && coef(mz)[1] == 0.0
    @test fit!(MNProbit(Xo, cho; intercepts = false, r = 0);
               maxiter = 100.0).converged
    # never-chosen alternative: no finite intercept MLE
    T = 100
    mu = @test_logs (:warn, r"never chosen") fit!(
        MNProbit(zeros(T, 3, 0), [isodd(t) ? 1 : 2 for t in 1:T]; r = 0);
        maxiter = 500)
    @test !mu.converged
    # contrast-unidentified mean design
    ch = vcat(fill(1, 30), fill(2, 50), fill(3, 40))
    Xa = zeros(120, 3, 1); Xa[:, 2, 1] .= 1.0
    @test_throws ArgumentError fit!(MNProbit(Xa, ch; r = 0))
    z = randn(Xoshiro(1), 120)
    Xc = repeat(reshape(z, 120, 1, 1), 1, 3, 1)
    @test_throws ArgumentError fit!(MNProbit(Xc, ch; intercepts = false, r = 0))
    xr = randn(Xoshiro(2), 120, 3, 1)
    @test_throws ArgumentError fit!(MNProbit(cat(xr, -3 .* xr; dims = 3), ch; r = 0))
    @test fit!(MNProbit(randn(Xoshiro(3), 120, 3, 2), ch; r = 0)).converged
end

# --- inference needs a maximum (#493) ----------------------------------
@testset "vcov/stderror refuse a saddle or an unconverged fit" begin
    X = reshape([0.5419522204102933 -0.3165954511658161 -0.32238911615896015
                 0.0971673186704572 -1.5259304065189514 1.1921661041016585],
                2, 3, 1)
    m = MNProbit(X, [3, 1]; intercepts = false, r = 1)
    fit!(m; maxiter = 0)
    @test !m.converged
    @test minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(-loglik_hessian(m)))) < 0
    for method in (:hessian, :opg, :sandwich)
        @test_throws ArgumentError vcov(m; method = method)
        @test_throws ArgumentError stderror(m; method = method)
    end
    # the override still refuses the indefinite information: the third
    # standard error used to be sqrt(max(-4.558, 0)) = 0
    @test_throws ArgumentError vcov(m; allow_unconverged = true)
    @test_throws ArgumentError stderror(m; allow_unconverged = true)
    @test_throws ArgumentError stderror(m; method = :sandwich,
                                        allow_unconverged = true)
    io = IOBuffer()
    show(io, MIME"text/plain"(), m)
    out = String(take!(io))
    @test occursin("did not converge", out)
    rows = filter(!isempty, split(out, "\n")[3:end])
    @test length(rows) == 3 && all(endswith.(rstrip.(rows), "-"))
end

# --- scalar D, loading rows, node columns, draw budget -----------------
@testset "MNP argument boundaries (#503, #462, #476, #478)" begin
    mu = zeros(2, 3)
    V = zeros(3, 1)
    a = choice_loglik_and_score(mu, V, [1, 2]; D = ones(3))
    b = choice_loglik_and_score(mu, V, [1, 2]; D = 1.0)
    @test a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
    c = choice_loglik_and_score(mu, V, [1, 2]; D = fill(2.0, 3))
    @test c[1] == choice_loglik_and_score(mu, V, [1, 2]; D = 2.0)[1]
    @test MP.ghk_choice_prob(zeros(3), V, 1; D = 1.0) ==
          MP.ghk_choice_prob(zeros(3), V, 1; D = ones(3))
    for bad in ([1.0, NaN, 1.0], [1.0, Inf, 1.0], [1.0, 0.0, 1.0],
                [1.0, -1.0, 1.0], 0.0, NaN)
        @test_throws ArgumentError choice_loglik_and_score(mu, V, [1, 2];
                                                           D = bad)
    end
    @test_throws DimensionMismatch choice_loglik_and_score(mu, V, [1, 2];
                                                           D = ones(2))
    # a surplus loading row switched GH -> Halton (0.654 -> 0.637)
    mu4 = reshape([2.29264426, 0.64388188, -0.17180493, 3.74907237], 1, :)
    V4 = [-0.0107529507 1.80970562; -0.6565907660 0.00168241106;
          0.3133862710 0.0763054062; 0.3539574460 -1.88769344]
    ll = choice_loglik_and_score(mu4, V4, [4]; D = ones(4))[1]
    @test abs(exp(ll) - 0.65440808) < 1e-6        # python reference, tilted rule
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu4, vcat(V4, [1.0 1.0]), [4]; D = ones(4))
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu4, V4[1:3, :], [4]; D = ones(4))
    # custom nodes: exactly r + 1 columns, one row per weight
    mu2 = zeros(1, 2)
    V2 = reshape([0.0, 1.0], 2, 1)
    ok = choice_loglik_and_score(mu2, V2, [1]; D = ones(2),
                                 nodes = ([0.0 0.0], [1.0]))
    @test ok[1] ≈ log(0.5)
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu2, V2, [1]; D = ones(2), nodes = ([0.0 10.0 0.0], [1.0]))
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu2, V2, [1]; D = ones(2), nodes = (reshape([0.0], 1, 1), [1.0]))
    @test_throws DimensionMismatch choice_loglik_and_score(
        mu2, V2, [1]; D = ones(2), nodes = ([0.0 0.0], [0.5, 0.5]))
    @test_throws ArgumentError choice_loglik_and_score(
        mu2, V2, [1]; D = ones(2), nodes = ([NaN 0.0], [1.0]))
    # a zero draw budget averaged an empty set: NaN
    Z = zeros(2, 0)
    @test_throws ArgumentError MP.ghk_choice_prob([0.0, 0.0], Z, 1;
                                                  D = ones(2), r_draws = 0)
    @test_throws ArgumentError MP.ghk_loglik(zeros(1, 2), Z, [1]; r_draws = 0)
    mg = MNProbit(zeros(2, 2, 1), [1, 2]; intercepts = false)
    th = copy(mg.theta)
    @test_throws ArgumentError fit!(mg; method = :ghk, r_draws = 0)
    @test mg.theta == th && isnan(mg.loglik)
    @test MP.ghk_choice_prob([0.0, 0.0], Z, 1; D = ones(2), r_draws = 1) ≈ 0.5
end

@testset "round 3: tilt, blend, symmetry, weights, separation, chart" begin
    # #352: a less noisy rival (D = 0.1) was 3.3% off; exact 0.367756
    p = exp(choice_loglik_and_score(zeros(1, 3), zeros(3, 1), [2];
                                    D = [0.1, 1.0, 1.0])[1])
    @test abs(p / 0.3677556825410359 - 1) < 2e-4
    # #128: mu = [g, 0, 1] with no factor, exact by a dense 1-D integral
    for (g, ex) in ((-8.0, -28.888414298356473), (-20.0, -146.579926526795))
        ll = choice_loglik_and_score(reshape([g, 0.0, 1.0], 1, 3),
                                     zeros(3, 1), [1])[1]
        @test abs(ll - ex) < 1e-6
    end
    # #345: a zero-padded binary loading is the same model, closed form
    g, a = -7.5, 4.3
    l1 = choice_loglik_and_score(reshape([g, 0.0], 1, 2),
                                 reshape([0.0, a], 2, 1), [1])
    l2 = choice_loglik_and_score(reshape([g, 0.0], 1, 2),
                                 [0.0 0.0; a 0.0], [1])
    @test l1[1] == l2[1] && l1[2] == l2[2]
    @test abs(l1[1] - MP.logndtr(g / sqrt(2 + a^2))) < 1e-12
    mu3 = [0.3 -0.2 0.5; -1.0 0.4 0.1]
    V3 = [0.2; -0.5; 0.4;;]
    a3 = choice_loglik_and_score(mu3, V3, [1, 3])
    b3 = choice_loglik_and_score(mu3, hcat(V3, zeros(3)), [1, 3])
    @test abs(a3[1] - b3[1]) < 1e-12 && all(b3[3][:, 2] .== 0)
    # #263: node weights are relative, finite, non-negative
    F = [0.0 -2.0; 0.0 2.0]
    base = choice_loglik_and_score(zeros(1, 2), zeros(2, 1), [1];
                                   nodes = (F, [0.5, 0.5]))
    for W in ([5.0, 5.0], [1e308, 1e308])
        o = choice_loglik_and_score(zeros(1, 2), zeros(2, 1), [1];
                                    nodes = (F, W))
        @test o[1] ≈ base[1] && o[2] ≈ base[2]
    end
    for W in ([-1.0, 2.0], [0.0, 0.0], [NaN, 1.0], [Inf, 1.0])
        @test_throws ArgumentError choice_loglik_and_score(
            zeros(1, 2), zeros(2, 1), [1]; nodes = (F, W))
    end
    # #420: no jump at the family switch, no cusp at tied rows
    a0 = 3 * sqrt(2.0)
    out = [choice_loglik_and_score(reshape([-3.0, 0.0, -60.0], 1, 3),
                                   [0.0; aa; 0.0;;], [1]) for aa in
           (a0 * (1 - 1e-10), a0 * (1 + 1e-10))]
    @test abs(out[1][1] - out[2][1]) < 1e-8
    @test abs(out[1][3][2, 1] - out[2][3][2, 1]) < 1e-6
    at = 3 / sqrt(2)
    Vt = [-at; 0.0; at;;]
    mut = [-2.0 0.4 1.1; 0.2 -1.0 0.7]
    ll, _, dV = choice_loglik_and_score(mut, Vt, [1, 2])
    h = 1e-7
    for i in 1:3
        vp = copy(Vt); vp[i, 1] += h
        vm = copy(Vt); vm[i, 1] -= h
        right = (choice_loglik_and_score(mut, vp, [1, 2])[1] - ll) / h
        left = (ll - choice_loglik_and_score(mut, vm, [1, 2])[1]) / h
        @test abs(right - left) < 1e-5
        @test abs(dV[i, 1] - (left + right) / 2) < 1e-5
    end
    # #605: factor signs and order. The sharp branch is a plain Halton
    # cloud again (owner decision, PR #719 / 2026-10-14: the symmetrised
    # orbit was 3-5x less accurate and cost the Fishing fit 3.4 nats), so
    # there the invariance holds to the cloud's noise (measured max moves
    # 0.0019 / 0.0039 / 0.0022); the Gauss-Hermite branch keeps it exactly.
    mu5 = reshape([0.14067943910283115, -0.1688419276279345,
                   0.8188357505677056, -0.7906732620426022], 1, 4)
    V5 = [-4.44796743024364 2.0653322851949887;
          1.5404628475279805 -3.7535504592745266;
          6.505591205368869 0.2331481770537056;
          -3.5980866226532098 1.4550699970258316]
    pr(A) = [exp(choice_loglik_and_score(mu5, A, [k])[1]) for k in 1:4]
    p0 = pr(V5)
    for A in (V5 .* [-1.0 1.0], V5 .* [1.0 -1.0], V5[:, [2, 1]])
        @test maximum(abs.(pr(A) .- p0)) < 8e-3
    end
    Vg = V5 ./ 4
    q0 = pr(Vg)
    for A in (Vg .* [-1.0 1.0], Vg .* [1.0 -1.0], Vg[:, [2, 1]])
        @test maximum(abs.(pr(A) .- q0)) < 1e-12
    end
    Fs, _ = MultinomialProbit.nodes_for_likelihood(2; sharp = 5.0)
    @test Fs == MultinomialProbit._halton_normal(3, 2^10)[1]
    @test maximum(abs.(p0 ./ sum(p0) .-
                       [0.3610138, 0.1938048, 0.4039584, 0.0412231])) < 3e-3
    # #581: one alternative is no choice
    @test_throws ArgumentError MNProbit(reshape(collect(0.0:7.0), 8, 1, 1),
                                        ones(Int, 8); intercepts = false, r = 0)
    # #642: separation flagged, not converged
    Ts = 100
    Xs = zeros(Ts, 2, 1)
    chs = zeros(Int, Ts)
    Xs[1:50, :, 1] .= [1.0 -1.0]; chs[1:50] .= 1
    Xs[51:end, :, 1] .= [-1.0 1.0]; chs[51:end] .= 2
    ms = @test_logs (:warn, r"separated") fit!(MNProbit(Xs, chs;
                                               intercepts = false, r = 0);
                                               maxiter = 500)
    @test !ms.converged && ms.separation ≈ [1.0]
    rng = Xoshiro(7)
    Xn = randn(rng, 200, 3, 2)
    chn = [argmax(Xn[t, :, 1] .* 0.6 .+ randn(rng, 3)) for t in 1:200]
    @test MP._separation_direction(Xn, chn) === nothing
    chq = [argmax(Xn[t, :, 1]) for t in 1:200]
    d = MP._separation_direction(Xn, chq)
    @test d !== nothing && d[1] > 0.99
    # #376: both normalisations and the mass defect
    mn = fit!(MNProbit(Xn, chn; r = 1))
    raw = predict_proba(mn; normalize = false)
    @test maximum(abs.(sum(predict_proba(mn), dims = 2) .- 1)) < 1e-12
    @test mass_defect(mn) ≈ 1 .- vec(sum(raw, dims = 2))
    @test mn.mass_defect ≈ maximum(abs, mass_defect(mn)) && mn.mass_defect < 1e-3
    # #645: a zero pivot makes the chart singular; vcov refuses
    T6, J6 = 400, 4
    X6 = Array{Float64}(undef, T6, J6, 1)
    for t in 1:T6, j in 1:J6
        X6[t, j, 1] = 0.7sin(0.13t + 0.7j) + 0.2cos(0.071t * j)
    end
    ch6 = [1 + mod(37t, J6) for t in 1:T6]
    m6 = MNProbit(X6, ch6; intercepts = false, r = 2)
    m6.theta .= [1.0, 0.0, 0.65, -0.25, 0.35, 0.80]
    m6.beta, m6.V = MP._unpack(m6, m6.theta)
    m6.converged = true
    @test_throws ArgumentError vcov(m6; method = :opg)
    @test all(isfinite, vcov(m6; method = :opg, allow_singular_chart = true))
    m6.theta[2] = 0.2
    m6.beta, m6.V = MP._unpack(m6, m6.theta)
    @test all(isfinite, vcov(m6; method = :opg))
end

@testset "GHK likelihood ignores observation row order (#619)" begin
    mu = [5.671347349259726 0.7030680274166967 -0.5542175268217469 0.42845918505707187 -0.6072857954627274 2.226262047348098
          -1.2556641311588315 1.4926790751025314 0.35972560246545665 1.4729169780465117 -5.49977946129139 2.6228400691944653]
    choice = [4, 1]
    V = [0.0 0.0
         -3.261049037785404 0.0
         -3.5258462254859224 -2.503124070275932
         -3.868636725222654 -3.848344435795788
         4.279705429237719 3.627916479526973
         3.3817943464273412 2.8854641918855872]
    ll1 = MP.ghk_loglik(mu, V, choice; r_draws = 1000, seed = 9)
    ll2 = MP.ghk_loglik(mu[[2, 1], :], V, choice[[2, 1]]; r_draws = 1000,
                        seed = 9)
    @test ll1 == ll2
    # and over a larger sample, any permutation
    rng = Xoshiro(3)
    M = randn(rng, 12, 4)
    ch = rand(rng, 1:4, 12)
    Vs = 0.5 .* randn(rng, 4, 1)
    perm = randperm(rng, 12)
    @test MP.ghk_loglik(M, Vs, ch; r_draws = 200) ≈
          MP.ghk_loglik(M[perm, :], Vs, ch[perm]; r_draws = 200) atol = 1e-10
end

@testset "GHK log-likelihood keeps its tail slope (#270)" begin
    # flat at log(1e-300) = -690.78 below gap -50 before; references are a
    # 1-d conditional-normal quadrature in log space
    V = reshape([0.3, -0.2, 0.1], 3, 1)
    for (g, ref) in ((-10.0, -37.51799189156518), (-55.0, -952.7976771047145),
                     (-60.0, -1131.5948698859718), (-80.0, -2001.2502944154676))
        ll = MP.ghk_loglik(reshape([g, 0.0, 0.5], 1, 3), V, [1]; r_draws = 4000)
        @test abs(ll - ref) < 0.05
    end
    # binary: one truncation step, exact
    @test MP.ghk_choice_logprob([0.0, 53.0], zeros(2, 0), 1; r_draws = 4) ≈
          MP.logndtr(-53 / sqrt(2)) rtol = 1e-10
    for x in (-40.0, -60.0, -200.0)
        @test MP._ndtri_log(MP.logndtr(x)) ≈ x rtol = 1e-10
    end
end

@testset "own-noise order: 7 below J = 6, 11 from J = 6 (#173)" begin
    @test [MultinomialProbit.default_qz(J) for J in (3, 5, 6, 10)] == [7, 7, 11, 11]
    for (J, r) in ((4, 1), (5, 2))
        mu = reshape(collect(range(-1.0, 1.0, length = J)), 1, J)
        V = 0.4 .* reshape(sin.(1:J*r), J, r)
        @test choice_loglik_and_score(mu, V, [2])[1] ==
              choice_loglik_and_score(mu, V, [2]; Qz = 7)[1]
    end
    mu6 = reshape(collect(range(-1.0, 1.0, length = 6)), 1, 6)
    V6 = 0.4 .* reshape(sin.(1:6), 6, 1)
    @test choice_loglik_and_score(mu6, V6, [2])[1] ==
          choice_loglik_and_score(mu6, V6, [2]; Qz = 11)[1]
    # ten independent probit runners: 1-d quadrature reference
    mu10 = reshape(collect(range(-2.0, 2.0, length = 10)), 1, 10)
    ll = choice_loglik_and_score(mu10, zeros(10, 0), [6])[1]
    @test abs(ll - (-3.514068769353895)) < 1e-5
    @test abs(choice_loglik_and_score(mu10, zeros(10, 0), [6]; Qz = 7)[1] -
              (-3.514068769353895)) > 5e-5
end

@testset "tilt curvature in the far tail; logndtr crossover (#716)" begin
    # the Fishing fit's first BFGS trial step: h was -196, a DomainError
    c0 = [-174823.91047091805, -174848.3798915476, -174803.45321830967]
    tj = [1.6888473976942278, 1.694115546292384, 2.9793037849297694]
    m, h, dm_c, dh_c, dm_t, dh_t = MP._tilt_mode(c0, tj, 1.0)
    @test h >= 1 && isfinite(m)
    @test all(isfinite, vcat(dm_c, dh_c, dm_t, dh_t))
    rng = Xoshiro(1)
    for sc in (1e3, 1e5, 1e6, 1e8)
        for _ in 1:200
            c = -abs.(sc .+ sc / 10 .* randn(rng, 3))
            @test MP._tilt_mode(c, 1 .+ 2 .* rand(rng, 3), 1.0)[2] >= 1
        end
    end
    for a in (-1e3, -5.7e4, -1e6)
        y = -a
        lam, gap = MP._mills_gap(a)
        @test gap ≈ 1 / y - 2 / y^3 + 10 / y^5 - 74 / y^7 rtol = 1e-13
        @test 0 < lam * gap <= 1
    end
    # the cephes / continued-fraction crossover moved to -20: both sides
    # agree there, and with the 60-term fraction it used to take from -3.54
    cfval(z) = log(0.5 * MP._erfcx(-z / sqrt(2))) - 0.5 * z * z
    for z in (-3.6, -5.0, -10.0, -19.9, -20.0, -20.1, -37.0, -1e4)
        @test MP.logndtr(z) ≈ cfval(z) rtol = 1e-14
    end
    for x in (14.2, 30.0, 1e3, 1e7)
        @test MP._erfcx_cf(x, 12) == MP._erfcx_cf(x, 60)
    end
end

@testset "Hessian vcov is covariate-unit equivariant (#702)" begin
    # l(b) = log Phi(a b / sqrt 2) + log Phi(-a b / sqrt 2): MLE b = 0,
    # exact Hessian -2 a^2 / pi, so a^2 vcov = pi / 2 at every unit
    for a in (1e-6, 1.0, 1e8)
        X = zeros(2, 2, 1)
        X[:, 1, 1] .= a
        m = MNProbit(X, [1, 2]; intercepts = false, r = 0)
        fit!(m)
        @test a^2 * vcov(m; method = :hessian)[1, 1] ≈ pi / 2 rtol = 1e-6
    end
end

@testset "sharp cloud keeps its own-noise resolution (#694)" begin
    # the opt-in symmetrised helper (the likelihood's sharp branch is plain)
    for r in 1:5
        F, W = MultinomialProbit._symmetric_halton(r, 1, 2^10)
        @test length(unique(F[:, r + 1])) >= 2^10 ÷ 8
        @test sum(W) ≈ 1.0
    end
    mu = reshape([0.6094341595088627, -2.079968212480991,
                  1.5009023916129145, 1.8811294327824277,
                  -3.902070377307673], 1, 5)
    V = [-2.7227514556621832 0.2673038870461732 -0.6612375437551032 -0.03512985407199214;
         -1.783645483057296 1.8387496528290126 1.6262996869719497 0.13806481898674158;
         2.356969696558986 0.9775240168074725 -1.7967106622591427 0.771027902717065;
         -2.004945542037578 1.8367681444891217 -0.10439101989911242 -0.386532169172916;
         -1.4237683041845364 2.5562345221532174 -0.3231085806743851 -0.8955986445411943]
    ref = [0.17760492, 0.02211342, 0.43580930, 0.36438589, 0.00008646]
    raw = [exp(choice_loglik_and_score(mu, V, [k])[1]) for k in 1:5]
    @test abs(sum(raw) - 1) < 0.02
    @test maximum(abs.(raw ./ sum(raw) .- ref)) < 0.01
end
