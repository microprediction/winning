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

# --- hard-constrained GMRFs are refused, not repriced (#804) -----------
#
# precision_matrix of a ConstrainedGMRF (and of a constrained
# WorkspaceGMRF) is the unconstrained base precision while mean() is the
# constrained mean, so the generic conversion priced X1, X2 iid N(0,1)
# for the law X1 == X2: max_cdf(0) came back 0.25 (exact 0.5) and
# expected_max 0.564 (exact 0). The extension now refuses these types.
@testset "hard-constrained GMRFs are refused (#804)" begin
    using GaussianMarkovRandomFields
    GMF = GaussianMarkovRandomFields
    Q2 = sparse([1.0 0.0; 0.0 1.0])
    base = GMF.GMRF(zeros(2), Q2)
    A, e = [1.0 -1.0], [0.0]
    cg = GMF.ConstrainedGMRF(base, A, e)
    wg = GMF.WorkspaceGMRF(zeros(2), Q2, GMF.GMRFWorkspace(Q2), A, e)
    for g in (cg, wg)
        @test_throws ArgumentError GaussMarkovChain(g)
        @test_throws ArgumentError max_cdf(g, 0.0)
        @test_throws ArgumentError excursion_probability(g, 0.0)
        @test_throws ArgumentError first_passage(g, 0.0)
        @test_throws ArgumentError argmax_marginals(g)
        @test_throws ArgumentError expected_max(g)
    end
    msg = sprint(showerror, try max_cdf(cg, 0.0) catch err; err end)
    @test occursin("hard linear constraints", msg)

    # the unconstrained base, plain or workspace-backed, still prices
    # identically to the chain built from its precision
    ref = chain_from_precision(zeros(2), Q2)
    wg0 = GMF.WorkspaceGMRF(zeros(2), Q2)
    for g in (base, wg0)
        @test max_cdf(g, 0.0) === max_cdf(ref, 0.0)
        @test first_passage(g, 0.0) == first_passage(ref, 0.0)
        @test expected_max(g) === expected_max(ref)
    end
    @test abs(max_cdf(base, 0.0) - 0.25) < 1e-8
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

@testset "scale equivariance up to the top of the range (#363)" begin
    # pad * sd used to overflow: max_cdf 0 at sd 1e308, argmax failed at 1e300
    huge = GaussMarkovChain([0.0], 1e308, Float64[], Float64[])
    @test abs(max_cdf(huge, 1e308) - GE.ndtr(1.0)) < 1e-12
    @test abs(excursion_probability(huge, 1e308) - GE.ndtr(-1.0)) < 1e-12
    @test maximum(abs.(first_passage(huge, 1e308) .-
                       [GE.ndtr(-1.0), GE.ndtr(1.0)])) < 1e-12
    @test argmax_marginals(huge) == [1.0]
    @test abs(expected_max(huge) / 1e308) < 1e-12
    ref = GaussMarkovChain([0.0, 0.3, -0.2], 1.0, [0.7, 0.5], [0.6, 0.9])
    mref = max_cdf(ref, 0.8)
    vref = max_cdf(ref, [0.8])[1]           # unaligned batch lattice
    aref = argmax_marginals(ref)
    xref = expected_max(ref)
    for a in (1e300, 1e305, 1e307, 1e308)
        c = GaussMarkovChain(a .* ref.mu, a * ref.sd0, ref.phi, a .* ref.s)
        @test abs(max_cdf(c, a * 0.8) - mref) < 1e-9
        @test abs(max_cdf(c, [a * 0.8])[1] - vref) < 1e-9
        @test abs(excursion_probability(c, a * 0.8) - (1 - mref)) < 1e-9
        @test maximum(abs.(argmax_marginals(c) .- aref)) < 1e-9
        @test abs(expected_max(c) / a - xref) < 1e-9
    end
    # a power-of-two rescaling is exact: the answers are the same bits
    for k in (-600, -3, 5, 600)
        a = ldexp(1.0, k)
        c = GaussMarkovChain(a .* ref.mu, a * ref.sd0, ref.phi, a .* ref.s)
        @test max_cdf(c, a * 0.8) == mref
        @test argmax_marginals(c) == aref
    end
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
    # Bool <: Real and true == 1.0: tol = true dropped the 0.5 coupling
    # and priced 0.125 for 0.1049 (#562 residual)
    for tol in (true, false)
        @test_throws ArgumentError chain_from_precision(zeros(3), Q; tol = tol)
        @test_throws ArgumentError chain_from_precision(zeros(3), Qr; tol = tol)
    end
    # numeric tolerances are unchanged, of any Real type
    for tol in (1e-12, 1e-10, big"1e-12", 1 // 10^12, 1f-6)
        ct = chain_from_precision(zeros(3), Qr; tol = tol)
        @test ct.phi == c.phi && ct.s == c.s && ct.sd0 == c.sd0
    end
    @test length(chain_from_precision(zeros(3), [2.0 -0.5 0.0; -0.5 2.0 -0.5;
                                                 0.0 -0.5 2.0]; tol = 0)) == 3
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
    # streaming is the same arithmetic as building every transition
    cs = stationary_ar1(0.6, 12)
    R = GE._offsets(cs, [0.7])
    lat = GE._lattice(cs, 120, R)
    cells(t) = GE._cells!(zeros(lat.L), lat, t)
    h1 = lat.dz[1] / 2
    v = [GE._interval_mass(zi - h1, min(zi + h1, R[1, 1]), 0.0, cs.sd0)
         for zi in cells(1)]
    for t in 1:11
        M = GE._transition!(zeros(lat.L, lat.L), cs.phi[t], cs.s[t],
                            cells(t), lat.dz[t], cells(t + 1), lat.dz[t + 1])
        keep = clamp.((R[t + 1, 1] .- (cells(t + 1) .- lat.dz[t + 1] / 2)) ./
                      lat.dz[t + 1], 0.0, 1.0)
        v = keep .* (M * v)
    end
    @test abs(GE._restricted_masses(cs, lat, R)[1][end, 1] - sum(v)) < 1e-14
    # an explicit lattice past the one-matrix budget is refused up front
    @test_throws ArgumentError max_cdf(stationary_ar1(0.5, 3), 0.0;
                                       points = 10_000)
    @test_throws ArgumentError argmax_marginals(stationary_ar1(0.5, 3);
                                                points = 10_000)
    # L * L wrapped: 2^32 squared to 0 and typemax(Int) to 1, so the
    # guard passed; it is checked without the overflow now (#533)
    for L in (1 << 32, typemax(Int), (1 << 31) + 1, 3037000500)
        @test_throws ArgumentError GE._check_lattice(L, 2)
    end
    @test GE._check_lattice(typemax(Int), 1) === nothing
    # the budget boundary itself is where it was
    @test GE._check_lattice(4472, 2) === nothing
    @test GE._check_lattice(isqrt(GE._LATTICE_DOUBLES), 1000) === nothing
    @test_throws ArgumentError GE._check_lattice(4473, 2)
    msg = sprint(showerror, try GE._check_lattice(4473, 2) catch err; err end)
    @test occursin("160.1 MB each", msg)
    # a single node has no transition, so no matrix to budget
    @test isfinite(max_cdf(GaussMarkovChain([0.0], 1.0, Float64[], Float64[]),
                           0.0; points = 10_000))
end

@testset "a chain and its exact reversal agree (#640)" begin
    # sd(X1) = 0.01 next to sd(X3) = 8: one shared lattice put X1 in a
    # single cell and gave 0.231 forward, 0.130 reversed
    c = GaussMarkovChain(zeros(3), 0.01, [2.0, -1 / 3], [0.28, 8.0])
    S = [0.0001 0.0002 -0.00006666666666666667
         0.0002 0.0788 -0.02626666666666667
         -0.00006666666666666667 -0.02626666666666667 64.00875555555556]
    p32 = S[2, 3] / S[3, 3]
    p21 = S[1, 2] / S[2, 2]
    cr = GaussMarkovChain(zeros(3), sqrt(S[3, 3]), [p32, p21],
                          [sqrt(S[2, 2] - p32^2 * S[3, 3]),
                           sqrt(S[1, 1] - p21^2 * S[2, 2])])
    sd = sqrt.([S[i, i] for i in 1:3])
    Rho = S ./ (sd * sd')
    exact = 1 / 8 + (asin(Rho[1, 2]) + asin(Rho[1, 3]) + asin(Rho[2, 3])) / (4pi)
    for points in (400, 800)
        f, r = max_cdf(c, 0.0; points), max_cdf(cr, 0.0; points)
        @test abs(f - exact) < 1e-5
        @test abs(r - exact) < 1e-5
        @test abs(f - r) < 1e-6
    end
    # the narrow node in the middle, and at either end of a longer chain
    for sds in ([1.0, 0.01, 1.0], [0.01, 1.0, 5.0, 5.0], [5.0, 5.0, 1.0, 0.01])
        n = length(sds)
        phis = [0.5 * sds[t + 1] / sds[t] for t in 1:(n - 1)]
        ss = [sqrt(0.75) * sds[t + 1] for t in 1:(n - 1)]
        ch = GaussMarkovChain(zeros(n), sds[1], phis, ss)
        # reverse: corr(t, t+1) = 0.5 throughout, so the reversed chain
        # has the same form read backwards
        rs = reverse(sds)
        chr = GaussMarkovChain(zeros(n), rs[1],
                               [0.5 * rs[t + 1] / rs[t] for t in 1:(n - 1)],
                               [sqrt(0.75) * rs[t + 1] for t in 1:(n - 1)])
        @test abs(max_cdf(ch, 0.0) - max_cdf(chr, 0.0)) < 1e-5
    end
end

@testset "a common shift moves nothing but the location (#657)" begin
    # means and threshold on a quarter grid, so every shifted input up
    # to 2^50 (where the ulp is 0.25) is exact and the law is unchanged
    z = GaussMarkovChain([0.0, 0.25, -0.25], 1.0, [0.7, 0.5], [0.6, 0.9])
    m0 = max_cdf(z, 0.5)
    e0 = excursion_probability(z, 0.5)
    f0 = first_passage(z, 0.5)
    a0 = argmax_marginals(z)
    x0 = expected_max(z)
    for shift in (1e12, 1e15, 2.0^50)
        c = GaussMarkovChain(z.mu .+ shift, z.sd0, z.phi, z.s)
        @test abs(max_cdf(c, 0.5 + shift) - m0) < 1e-12
        @test abs(excursion_probability(c, 0.5 + shift) - e0) < 1e-12
        f = first_passage(c, 0.5 + shift)
        @test abs(sum(f) - 1) < 1e-12
        @test maximum(abs.(f .- f0)) < 1e-12
        @test maximum(abs.(argmax_marginals(c) .- a0)) < 1e-9
        @test abs(expected_max(c) - (x0 + shift)) <= 4 * eps(shift) + 1e-9
    end
    # the issue's one-node reproducer, where the threshold offset is exact
    one = GaussMarkovChain([2.0^50], 1.0, Float64[], Float64[])
    @test abs(max_cdf(one, 2.0^50) - 0.5) < 1e-12
    @test abs(excursion_probability(one, 2.0^50) - 0.5) < 1e-12
    @test abs(sum(first_passage(one, 2.0^50)) - 1) < 1e-12
    @test argmax_marginals(one) == [1.0]
end

@testset "reciprocal finite scales do not square to NaN (#663)" begin
    for (phi, sd0) in ((1e160, 1e-160), (1e200, 1e-200))
        c = GaussMarkovChain([0.0, 0.0], sd0, [phi], [1.0])
        @test GE.marginal_sd(c) ≈ [sd0, sqrt(2.0)]
        @test abs(max_cdf(c, 0.0) - 0.375) < 1e-4
        @test maximum(abs.(first_passage(c, 0.0) .- [0.5, 0.125, 0.375])) < 1e-4
    end
    # a marginal scale genuinely past Float64 is refused, not gridded
    big = GaussMarkovChain([0.0, 0.0], 1e200, [1e200], [1.0])
    @test_throws ArgumentError max_cdf(big, 0.0)
end

@testset "no lattice warning for a central threshold (#685)" begin
    n = 200
    c = stationary_ar1(0.5, n)
    @test_logs max_cdf(c, 2.0)
    @test_logs first_passage(c, 3.0)
    @test_logs max_cdf(stationary_ar1(0.5, 20), 0.5; points = 2000)
    # a threshold that does force a much coarser lattice still warns
    @test_logs (:warn, r"far outside") max_cdf(c, 30.0)
end

# --- opposite-sign finite means: scale, then subtract (#657, #363) -----
#
# The means and thresholds were subtracted before the power-of-two
# rescaling, so 1e308 - (-1e308) overflowed to Inf although the contrast
# is finite in the chain's own units: max_cdf priced the second node as
# unbounded (0.5 for Phi(0) Phi(2) = 0.48862), and expected_max and
# argmax_marginals threw on internally generated infinities.
@testset "opposite-sign finite means (#657, #363)" begin
    Phi = GE.ndtr
    c = GaussMarkovChain([1e308, -1e308], 1e308, [0.0], [1e308])
    exact = Phi(0.0) * Phi(2.0)                     # 0.4886249340259104
    @test abs(max_cdf(c, 1e308) - exact) < 1e-12
    @test abs(excursion_probability(c, 1e308) - (1 - exact)) < 1e-12
    @test abs(sum(first_passage(c, 1e308)) - 1) < 1e-12
    @test abs(max_cdf(c, [1e308])[1] - exact) < 1e-5   # unaligned lattice
    # sd 1e307: the second node wins with probability Phi(-14.14) ~ 1e-45
    for (mu, top) in (([1e308, -1e308], 1), ([-1e308, 1e308], 2))
        c = GaussMarkovChain(mu, 1e307, [0.0], [1e307])
        x = expected_max(c)
        @test isfinite(x) && abs(x / 1e308 - 1) < 1e-6
        P = argmax_marginals(c)
        @test abs(P[top] - 1) < 1e-12 && 0 <= P[3 - top] < 1e-40
    end
    # sd 1: the contrast is past Float64 in the chain's units, so the
    # lower node is saturated; it can never be the maximum
    for (mu, top) in (([1e308, -1e308], 1), ([-1e308, 1e308], 2))
        c = GaussMarkovChain(mu, 1.0, [0.0], [1.0])
        @test argmax_marginals(c)[top] == 1.0
        @test argmax_marginals(c)[3 - top] == 0.0
        x = expected_max(c)
        @test isfinite(x) && abs(x / 1e308 - 1) < 1e-2   # threshold grid
        @test abs(max_cdf(c, 1e308) - 0.5) < 1e-12
    end
    # ordinary inputs: the scaled-first contrasts are the same bits as
    # the old subtract-then-scale ones, for e > 0 and e <= 0 alike
    for (mu, sd) in (([0.0, 0.3, -0.2], 1.0), ([3.0, -3.0, 1.7], 2.5),
                     ([-1e5, 2e5, 7.0], 1e4), ([0.1, -0.1, 0.05], 1e-3),
                     ([1e12, 1e12 + 0.25, 1e12 - 0.5], 3.0))
        ch = GaussMarkovChain(mu, sd, [0.6, -0.4], [0.8sd, 0.9sd])
        _, e = GE._unit_scaled(ch)
        us = [mu[1] + sd, mu[2] - 2sd, Inf, -Inf]
        @test GE._offsets(ch, us, e) == ldexp.(GE._offsets(ch, us), -e)
        a, d = GE._mean_contrasts(ch, e)
        @test a === mu[1]
        @test d == ldexp.(ch.mu .- mu[1], -e)
    end
end

println("all GMRFExtremes tests passed")
