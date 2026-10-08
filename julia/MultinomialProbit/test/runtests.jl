# Fixture parity (python winning.likelihood/mnprobit is the spec),
# score vs finite differences AND vs ForwardDiff duals, GHK-vs-exact
# agreement, and an end-to-end fit recovery.
#   julia --project=julia/MultinomialProbit/test julia/MultinomialProbit/test/runtests.jl
using Test
using Random: Xoshiro
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
# logndtr removes both. It does NOT remove the quadrature error
# underneath, and the last testset pins that rather than pretending
# otherwise.
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

    # recorded, not claimed fixed: 38% away at a gap of -20
    ll20, _, _ = binary(-20.0)
    ex20 = MP.logndtr(-20.0 / sqrt(2))
    @test abs(ll20 - ex20) / abs(ex20) > 0.3
    @test ll20 < ex20                     # understates the probability

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
    @test abs(exp(ll) - 0.65439858) < 1e-6
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
