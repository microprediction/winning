# Closed forms, a Monte Carlo referee (testing only, per the house
# rule), the measured laws of the winning repo's tridiagonal track,
# and the refusal contract.
#   julia --project=julia/GMRFExtremes julia/GMRFExtremes/test/runtests.jl
using Test
using Random: Xoshiro
using SparseArrays: spdiagm, sparse
using LinearAlgebra: SymTridiagonal, diag

using GMRFExtremes
const GE = GMRFExtremes

function stationary_ar1(phi, n; mu = 0.0)
    GaussMarkovChain(fill(mu, n), 1.0, fill(phi, n - 1),
                     fill(sqrt(1 - phi^2), n - 1))
end

function simulate(c::GaussMarkovChain, m, rng)
    n = length(c)
    X = zeros(m, n)
    X[:, 1] .= c.mu[1] .+ c.sd0 .* randn(rng, m)
    for t in 1:(n - 1)
        X[:, t + 1] .= c.mu[t + 1] .+ c.phi[t] .* (X[:, t] .- c.mu[t]) .+
                       c.s[t] .* randn(rng, m)
    end
    return X
end

@testset "iid closed form" begin
    c = stationary_ar1(0.0, 12)
    for u in (-0.5, 0.5, 1.5, 2.5)
        @test abs(max_cdf(c, u) - GE.ndtr(u)^12) < 2e-4
    end
end

@testset "chain from tridiagonal precision reproduces the chain" begin
    phi, n = 0.8, 15
    # stationary AR(1) precision, unit marginal variance
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ (1 - phi^2)
    o = fill(-phi / (1 - phi^2), n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    c2 = chain_from_precision(zeros(n), Q)
    c1 = stationary_ar1(phi, n)
    for u in (0.0, 1.0, 2.0)
        @test abs(max_cdf(c1, u) - max_cdf(c2, u)) < 1e-9
    end
    P1 = argmax_marginals(c1)
    P2 = argmax_marginals(c2)
    @test maximum(abs.(P1 .- P2)) < 1e-9
end

@testset "SymTridiagonal precision: the same chain in O(n)" begin
    rng = Xoshiro(7)
    n = 40
    d = 2.0 .+ rand(rng, n)
    e = 0.6 .* (rand(rng, n - 1) .- 0.5)          # diagonally dominant, SPD
    mu = randn(rng, n)
    Qs = SymTridiagonal(d, e)
    c_sym = chain_from_precision(mu, Qs)
    c_dense = chain_from_precision(mu, Matrix(Qs))
    c_sparse = chain_from_precision(mu, sparse(Qs))
    for c in (c_dense, c_sparse)
        @test abs(c.sd0 - c_sym.sd0) < 1e-12
        @test maximum(abs.(c.phi .- c_sym.phi)) < 1e-12
        @test maximum(abs.(c.s .- c_sym.s)) < 1e-12
    end
    # the chain's marginals are the precision's inverse diagonal
    sd_ref = sqrt.(diag(inv(Matrix(Qs))))
    @test maximum(abs.(GE.marginal_sd(c_sym) .- sd_ref)) < 1e-10
    # O(n): a chain the dense path could not allocate
    big = 200_000
    cbig = chain_from_precision(zeros(big), SymTridiagonal(fill(2.0, big), fill(-0.9, big - 1)))
    @test length(cbig) == big
    @test_throws ErrorException chain_from_precision(zeros(3), SymTridiagonal([1.0, 0.1, 1.0], [0.9, 0.9]))
end

@testset "precision route preserves index order" begin
    # a stationary chain's argmax law is symmetric, so it cannot detect
    # an orientation flip. Drift breaks the symmetry and pins it.
    n = 12
    phi, s2 = 0.7, 0.5
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ s2
    o = fill(-phi / s2, n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    mu = collect(0.25 .* (1:n))            # rising: late indices favoured
    c = chain_from_precision(mu, Q)
    P = argmax_marginals(c)
    @test P[n] > 5 * P[1]
    @test argmax(P) == n
end

@testset "Monte Carlo referee" begin
    rng = Xoshiro(1)
    mu = [0.0, 0.3, 0.6, 0.4, 0.0, -0.3, -0.1, 0.2, 0.5, 0.1]
    c = GaussMarkovChain(mu, 0.8, fill(0.7, 9),
                         vcat(fill(0.6, 4), fill(0.9, 5)))
    X = simulate(c, 400_000, rng)
    mx = vec(maximum(X, dims = 2))
    for u in (0.5, 1.5, 2.5)
        @test abs(max_cdf(c, u) - (sum(mx .<= u) / length(mx))) < 4e-3
    end
    @test abs(expected_max(c) - (sum(mx) / length(mx))) < 5e-3
    # argmax marginals
    am = argmax_marginals(c)
    counts = zeros(10)
    for r in 1:size(X, 1)
        counts[argmax(@view X[r, :])] += 1
    end
    @test maximum(abs.(am .- counts ./ size(X, 1))) < 5e-3
    # first passage at u = 1.0
    fp = first_passage(c, 1.0)
    @test abs(sum(fp) - 1) < 1e-9
    hit = [findfirst(>(1.0), @view X[r, :]) for r in 1:200_000]
    for t in (1, 3, 7)
        emp = sum(h -> h !== nothing && h == t, hit) / 200_000
        @test abs(fp[t] - emp) < 5e-3
    end
end

@testset "the measured laws of the tridiagonal track" begin
    # exp2: positive correlation piles the argmax onto the BOUNDARY --
    # measured ends/middle ~ 2.6 at phi = 0.9, n = 20
    P = argmax_marginals(stationary_ar1(0.9, 20); points = 400)
    ratio = (P[1] + P[end]) / 2 / P[10]
    @test 2.0 < ratio < 3.4
    @test abs(P[1] - P[end]) < 0.01          # exchange symmetry
    # exp3: the random walk's discrete arcsine U-shape
    n = 40
    w = GaussMarkovChain(zeros(n), 1.0, fill(1.0, n - 1),
                         fill(1.0, n - 1))
    Pw = argmax_marginals(w; points = 400)
    @test Pw[1] > 4 * Pw[div(n, 2)]
    @test Pw[end] > 4 * Pw[div(n, 2)]
end

@testset "refusal contract" begin
    n = 8
    Q = spdiagm(-2 => fill(0.1, n - 2), -1 => fill(-0.4, n - 1),
                0 => fill(1.5, n), 1 => fill(-0.4, n - 1),
                2 => fill(0.1, n - 2))
    @test_throws ErrorException chain_from_precision(zeros(n), Q)
end

@testset "GaussianMarkovRandomFields dispatch extension" begin
    # `using GaussianMarkovRandomFields` loads GMRFDispatchExt, so the
    # GMRF methods below are the extension's own, not local glue
    using GaussianMarkovRandomFields
    GMF = GaussianMarkovRandomFields
    phi, n = 0.8, 12
    d = vcat([1.0], fill(1 + phi^2, n - 2), [1.0]) ./ (1 - phi^2)
    o = fill(-phi / (1 - phi^2), n - 1)
    Q = spdiagm(-1 => o, 0 => d, 1 => o)
    g = GMF.GMRF(zeros(n), Q)
    ref = stationary_ar1(phi, n)
    for u in (0.5, 1.5)
        @test abs(max_cdf(g, u) - max_cdf(ref, u)) < 1e-8
    end
    am = argmax_marginals(g)
    @test abs(sum(am) - 1) < 1e-9
    @test abs(am[1] - am[end]) < 0.01
end

# --- a transition narrower than the lattice spacing (#244) -------------
#
# The kernel was discretised as density x spacing, which is a probability
# only where the density is flat across a cell. An innovation sd far
# below the spacing makes it a SPIKE between samples: with s = 1e-4 on a
# lattice spaced by the marginal sd, each column of the transition matrix
# summed to about 80, max_cdf returned 79.988 for a probability and the
# excursion probability came back as -78.99.
#
# The cell mass is a CDF difference with the variance deflated by
# dx^2/12, which is what the cell averaging adds. That leaves the
# resolved regime bit-for-bit as it was and gives the unresolved one its
# correct degenerate limit.
@testset "a transition narrower than the lattice" begin
    c = GaussMarkovChain([0.0, 0.0], 1.0, [1.0], [1e-4])

    # X2 = X1 + eps: bivariate normal, corr = 1/sqrt(1 + s^2), and
    # P(max <= 0) = 1/4 + asin(rho)/(2pi).
    rho = 1.0 / sqrt(1.0 + 1e-8)
    exact = 0.25 + asin(rho) / (2pi)

    for pts in (200, 400, 1600)
        m = max_cdf(c, 0.0; points = pts)
        @test 0.0 <= m <= 1.0
        @test abs(m - exact) < 1e-3
        e = excursion_probability(c, 0.0; points = pts)
        @test 0.0 <= e <= 1.0
        @test abs(m + e - 1.0) < 1e-12
        f = first_passage(c, 0.0; points = pts)
        @test all(0.0 .<= f .<= 1.0)
        @test abs(sum(f) - 1.0) < 1e-10
    end

    # every transition column is a probability distribution, which is the
    # property that failed: the columns summed to 80.
    x, dx = GE._grid(c, 400)
    T = GE._transitions(c, x, dx)
    for M in T
        colsums = vec(sum(M; dims = 1))
        @test maximum(abs.(colsums .- 1.0)) < 1e-9
    end

    # and the resolved regime is unchanged: 12 iid normals, closed form
    iid = GaussMarkovChain(zeros(12), 1.0, zeros(11), ones(11))
    for u in (-0.5, 0.5, 1.5, 2.5)
        @test abs(max_cdf(iid, u) - GE.ndtr(u)^12) < 2e-4
    end
end

# --- a threshold out in the tail (#197) --------------------------------
#
# The grid was built from the chain's means and marginal sds alone, so a
# threshold above the last cell left the occupancy identically one and
# every answer stopped depending on u: 9, 10 and 20 sd out all returned
# the same 1.11e-15, which is quadrature noise, not a tail. The
# excursion was then taken as 1 - max_cdf, a subtraction of two numbers
# that agree to every bit, so it went NEGATIVE at 10 sd, and
# first_passage differenced the same quantities and produced a negative
# passage mass.
#
# Now the grid covers the threshold, the exceedance is summed over the
# cells ABOVE u rather than subtracted from one, and the cell mass falls
# back to the midpoint density where ndtr has saturated.
@testset "a threshold out in the tail" begin
    c = GaussMarkovChain([0.0], 1.0, Float64[], Float64[])

    # one node is one standard normal, so the excursion is its tail.
    # Computed in log space, since the double-precision cdf saturates
    # around 8.3 sd and cannot express the answer directly.
    logtail(u) = -0.5 * u^2 - log(u) - 0.5 * log(2pi) +
                 log1p(-1 / u^2 + 3 / u^4)

    for u in (4.0, 6.0, 9.0, 12.0)
        e = excursion_probability(c, u)
        @test e > 0.0                      # was 0, or negative at 10 sd
        @test isfinite(e)
        @test abs(e - exp(logtail(u))) / exp(logtail(u)) < 5e-2
    end

    # the answer must DEPEND on the threshold, which is what failed
    @test excursion_probability(c, 9.0) > excursion_probability(c, 12.0)
    @test excursion_probability(c, 12.0) > excursion_probability(c, 15.0)

    # first passage is a distribution, with no negative entry
    for u in (6.0, 9.0, 12.0)
        f = first_passage(c, u; points = 200)
        @test all(f .>= 0.0)
        @test abs(sum(f) - 1.0) < 1e-12
        @test f[1] > 0.0
    end

    # a chain with transitions, so the grid extension has to carry them
    c3 = GaussMarkovChain(zeros(3), 1.0, fill(0.8, 2), fill(0.3, 2))
    for u in (2.0, 5.0, 9.0)
        e = excursion_probability(c3, u)
        @test 0.0 < e <= 1.0
        f = first_passage(c3, u; points = 200)
        @test all(f .>= 0.0)
        @test abs(sum(f) - 1.0) < 1e-10
    end
    @test excursion_probability(c3, 5.0) > excursion_probability(c3, 9.0)
end

# --- the chain's shape is enforced at construction (#347) --------------
@testset "GaussMarkovChain validates its shape" begin
    ok = GaussMarkovChain([0.0, 0.0], 1.0, [0.0], [1.0])
    @test length(ok) == 2
    # surplus transitions used to be accepted and silently ignored
    @test_throws DimensionMismatch GaussMarkovChain([0.0, 0.0], 1.0,
                                                    [0.0, 25.0], [1.0, 1e-12])
    @test_throws DimensionMismatch GaussMarkovChain([0.0], 1.0, [1e100], [1e-100])
    # short ones failed later with a raw BoundsError
    @test_throws DimensionMismatch GaussMarkovChain(zeros(3), 1.0, [0.5], [1.0])
    @test_throws DimensionMismatch GaussMarkovChain(zeros(3), 1.0, [0.5, 0.5], [1.0])
    @test_throws ArgumentError GaussMarkovChain(Float64[], 1.0, Float64[], Float64[])
    @test_throws ArgumentError GaussMarkovChain([0.0, 0.0], -1.0, [0.0], [1.0])
    @test_throws ArgumentError GaussMarkovChain([0.0, 0.0], 1.0, [0.0], [-1.0])
    @test_throws ArgumentError GaussMarkovChain([0.0, NaN], 1.0, [0.0], [1.0])
    @test_throws ArgumentError GaussMarkovChain([0.0, 0.0], 1.0, [Inf], [1.0])
    # integer input still converts
    @test GaussMarkovChain([0, 1], 1, [0], [1]).mu == [0.0, 1.0]
end

# --- an asymmetric precision is refused, not symmetrised (#356) --------
@testset "chain_from_precision requires symmetry" begin
    mu = zeros(2)
    Qsym = [2.0 -1.0; -1.0 2.0]
    Qlower = [2.0 0.0; -1.0 2.0]       # used to price as diag(2, 2)
    Qasym = [2.0 -1.0; -0.1 2.0]       # used to price as Qsym
    c = chain_from_precision(mu, Qsym)
    @test c.phi ≈ [0.5]
    @test abs(max_cdf(c, 0.0; points = 1600) - 1 / 3) < 2e-3
    for Q in (Qlower, Qasym)
        @test_throws ErrorException chain_from_precision(mu, Q)
        @test_throws ErrorException chain_from_precision(mu, sparse(Q))
    end
    # symmetric sparse and an explicit tolerance still pass
    @test chain_from_precision(mu, sparse(Qsym)).phi ≈ [0.5]
    @test chain_from_precision(mu, [2.0 -1.0; -1.0 + 1e-10 2.0];
                               tol = 1e-8).phi ≈ [0.5] atol = 1e-8
    @test_throws ErrorException chain_from_precision(mu, [2.0 NaN; NaN 2.0])
end

# --- a batch of thresholds is covered at both ends (#417) --------------
@testset "vector max_cdf agrees with the scalar calls" begin
    c = GaussMarkovChain([0.0], 1.0, Float64[], Float64[])
    for us in ([-9.0, 0.0], [-12.0, 0.0], [-12.0, 0.0, 12.0],
               [-3.0, -1.0, 0.5, 2.0])
        vb = max_cdf(c, us)
        vs = [max_cdf(c, u) for u in us]
        @test all(vb .> 0)                       # was exactly 0 at -9, -12
        @test all(abs.(vb .- vs) .<= 1e-3 .* vs)
        @test all(abs.(vb .- GE.ndtr.(us)) .<= 1e-3 .* GE.ndtr.(us))
        @test issorted(vb[sortperm(us)])
    end
    c3 = GaussMarkovChain(zeros(3), 1.0, fill(0.8, 2), fill(0.6, 2))
    us = [-4.0, 0.0, 2.0]
    @test maximum(abs.(max_cdf(c3, us) .- [max_cdf(c3, u) for u in us]) ./
                  [max_cdf(c3, u) for u in us]) < 1e-2
    @test max_cdf(c3, Float64[]) == Float64[]
end

# --- the lattice is scale-equivariant (#363) ---------------------------
@testset "scale equivariance below 1e-12" begin
    ref = GaussMarkovChain([0.0, 0.3, -0.2], 1.0, [0.7, 0.5], [0.6, 0.9])
    mref = max_cdf(ref, 0.8)
    eref = excursion_probability(ref, 0.8)
    fref = first_passage(ref, 0.8)
    aref = argmax_marginals(ref)
    xref = expected_max(ref)
    for a in (1e-16, 1e-14, 1e-9, 1e3, 1e8)
        c = GaussMarkovChain(a .* ref.mu, a * ref.sd0, ref.phi, a .* ref.s)
        @test abs(max_cdf(c, a * 0.8) - mref) < 1e-9
        @test abs(excursion_probability(c, a * 0.8) - eref) < 1e-9
        @test maximum(abs.(first_passage(c, a * 0.8) .- fref)) < 1e-9
        @test maximum(abs.(argmax_marginals(c) .- aref)) < 1e-9
        @test abs(expected_max(c) / a - xref) < 1e-9 * max(1, abs(xref))
    end
    # the one-node reproducer: 0.6249 against Phi(1) = 0.8413
    tiny = GaussMarkovChain([0.0], 1e-14, Float64[], Float64[])
    @test abs(max_cdf(tiny, 1e-14) - GE.ndtr(1.0)) < 1e-12
    @test abs(excursion_probability(tiny, 1e-14) - GE.ndtr(-1.0)) < 1e-12
end

# --- sub-cell scales keep their within-cell information (#244) ---------
@testset "unresolved scales: initial law, identity chain, drift" begin
    # a strictly positive but sub-cell sd0: was 5.5e-15 instead of 0.25
    c = GaussMarkovChain([0.0, 0.0], 1e-4, [0.0], [1.0])
    for pts in (200, 400, 1600)
        @test abs(max_cdf(c, 0.0; points = pts) - 0.25) < 1e-9
        @test abs(excursion_probability(c, 0.0; points = pts) - 0.75) < 1e-9
        @test maximum(abs.(first_passage(c, 0.0; points = pts) .-
                           [0.5, 0.25, 0.25])) < 1e-9
    end
    # an exactly repeated variable cannot first cross after step 1: the
    # boundary cell was re-clipped at every step (0.154 invented mass)
    n = 20
    rep = GaussMarkovChain(zeros(n), 1.0, ones(n - 1), zeros(n - 1))
    x, _ = GE._grid(rep, 20)
    for (u, pts) in ((x[10], 20), (x[10], 400), (0.3, 400))
        p = first_passage(rep, u; points = pts)
        @test abs(p[1] - (1 - GE.ndtr(u))) < 1e-9
        @test maximum(abs.(p[2:n])) < 1e-9
        @test abs(p[end] - GE.ndtr(u)) < 1e-9
    end
    # a sub-cell drift is carried, not snapped away at every step:
    # X_t = X_1 + 0.001 (t - 1) exactly, so max = X_200
    m = 200
    drift = GaussMarkovChain(0.001 .* (0:(m - 1)), 1.0, ones(m - 1), zeros(m - 1))
    @test abs(max_cdf(drift, 0.5; points = 300) - GE.ndtr(0.5 - 0.199)) < 2e-3
end

# --- argmax refuses an order the lattice cannot see (#432) -------------
@testset "argmax ordering below lattice resolution" begin
    Phi = GE.ndtr
    exact(phi, d, s) = Phi(-d / sqrt((1 - phi)^2 + s^2))
    # the reproducer returned [0.5, 0.5]; truth [Phi(-1), Phi(1)]
    for (d, s) in ((0.001, 0.001), (0.001, 0.0001))
        c = GaussMarkovChain([0.0, d], 1.0, [1.0], [s])
        @test_throws ErrorException argmax_marginals(c)
        @test_throws ErrorException argmax_marginals(c; points = 1600)
    end
    # refused at the default lattice, answered (correctly) once raised
    c = GaussMarkovChain([0.0, 0.05], 1.0, [1.0], [0.05])
    @test_throws ErrorException argmax_marginals(c)
    @test abs(argmax_marginals(c; points = 1600)[1] - exact(1.0, 0.05, 0.05)) < 2e-3
    # wherever it answers, it answers to the oracle
    for (phi, d, s) in ((1.0, 0.3, 0.3), (0.99, 0.05, 0.2), (1.0, 0.5, 1.0),
                        (1.0, 0.2, 0.5), (0.95, 0.1, 0.3), (0.9, 0.0, 0.05))
        c = GaussMarkovChain([0.0, d], 1.0, [phi], [s])
        P = argmax_marginals(c)
        @test abs(P[1] - exact(phi, d, s)) < 2e-3
    end
end

@testset "NaN thresholds are refused, infinities are limits (#513)" begin
    c1 = GaussMarkovChain([0.0], 1.0, Float64[], Float64[])
    c2 = GaussMarkovChain([0.0, 0.2], 1.0, [0.5], [0.8])
    for c in (c1, c2)
        @test_throws ArgumentError max_cdf(c, NaN)
        @test_throws ArgumentError excursion_probability(c, NaN)
        @test_throws ArgumentError first_passage(c, NaN)
        @test_throws ArgumentError max_cdf(c, [0.0, NaN])
        @test_throws ArgumentError excursion_probability(c, [0.0, NaN])
        @test abs(max_cdf(c, Inf) - 1) < 1e-12 && max_cdf(c, -Inf) == 0.0
        f = first_passage(c, 0.3)
        @test abs(sum(f) - 1) < 1e-12
        @test abs(max_cdf(c, 0.3) + excursion_probability(c, 0.3) - 1) < 1e-12
    end
end

@testset "lattice budgets below two are refused (#550)" begin
    c = GaussMarkovChain([0.0, 0.2], 1.0, [0.5], [0.8])
    for points in (0, 1, 2.5)
        @test_throws ArgumentError max_cdf(c, 0.0; points = points)
        @test_throws ArgumentError excursion_probability(c, 0.0; points = points)
        @test_throws ArgumentError first_passage(c, 0.0; points = points)
        @test_throws ArgumentError expected_max(c; points = points)
        @test_throws ArgumentError argmax_marginals(c; points = points)
    end
    for nu in (0, 1)
        @test_throws ArgumentError expected_max(c; nu = nu)
    end
    @test isfinite(max_cdf(c, 0.0; points = 2))
    @test isfinite(expected_max(c; nu = 2))
end

@testset "chain_from_precision tol must be finite and nonnegative (#562)" begin
    Q = [2.0 0.0 0.5; 0.0 2.0 0.0; 0.5 0.0 2.0]
    for tol in (Inf, NaN, -1e-3)
        @test_throws ArgumentError chain_from_precision(zeros(3), Q; tol = tol)
    end
    @test_throws ErrorException chain_from_precision(zeros(3), Q; tol = 0.1)
    Qr = [2.0 -0.5 1e-14; -0.5 2.0 -0.5; 1e-14 -0.5 2.0]
    c = chain_from_precision(zeros(3), Qr; tol = 1e-12)
    @test length(c) == 3
end

@testset "memory is O(L^2), not O(n L^2) (#533)" begin
    # 1000 nodes kept 999 dense 400 x 400 transitions, 1.28 GB under a
    # promised 160 MB; the passes now stream one transition
    n = 1000
    c = stationary_ar1(0.8, n)
    max_cdf(c, 3.0)
    bytes = @allocated max_cdf(c, 3.0)
    @test bytes < 20_000_000
    p = max_cdf(c, 3.0)
    @test 0 < p < 1
    # streaming is the same arithmetic as the stored transitions
    cs = stationary_ar1(0.6, 12)
    x, dx = GE._grid(cs, 120)
    T = GE._transitions(cs, x, dx)
    v = [GE._interval_mass(xi - dx / 2, min(xi + dx / 2, 0.7), 0.0, 1.0)
         for xi in x]
    keep = clamp.((0.7 .- (x .- dx / 2)) ./ dx, 0.0, 1.0)
    for M in T
        v = keep .* (M * v)
    end
    @test abs(GE._restricted_masses(cs, x, dx, [0.7])[1][end, 1] - sum(v)) <
          1e-14
    # an explicit lattice past the one-matrix budget is refused up front
    @test_throws ArgumentError max_cdf(stationary_ar1(0.5, 3), 0.0;
                                       points = 10_000)
    @test_throws ArgumentError argmax_marginals(stationary_ar1(0.5, 3);
                                                points = 10_000)
    # a single node has no transition, so no matrix to budget
    @test isfinite(max_cdf(GaussMarkovChain([0.0], 1.0, Float64[], Float64[]),
                           0.0; points = 10_000))
end

println("all GMRFExtremes tests passed")
