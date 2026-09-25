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

println("all GMRFExtremes tests passed")
