# GMRFExtremes.jl: the order-statistic layer the graphical-model
# stack does not compute. A Gauss-Markov chain's filter/smoother gives
# marginals, joints and samples; it does NOT give P(X_t is the max),
# the max-CDF, or the first-passage law. This package computes those
# EXACTLY (to quadrature) by restricted transfer-operator passes on
# the chain -- the machinery measured in research/tridiagonal exp1-exp3
# of the winning repo, generalized off stationary AR(1) to any
# Gauss-Markov chain, including any GMRF whose precision is
# TRIDIAGONAL (the chain is read off the precision's bidiagonal
# Cholesky). In the R world this query class belongs to Bolin &
# Lindgren's `excursions` (JRSS-B 2015), by sequential importance
# sampling; here the chain case is deterministic and exact.
#
# THE REFUSAL CONTRACT (the winning repo's -fast convention): only
# structures the method treats exactly are accepted -- a tridiagonal
# precision, or explicit chain parameters. Wider bandwidth and general
# sparsity are refused with a pointer, not approximated silently.
# Loading GaussianMarkovRandomFields.jl arms method dispatch on its
# GMRF type via a package extension.
module GMRFExtremes

using LinearAlgebra: cholesky, Symmetric, diag
using SparseArrays: SparseMatrixCSC, findnz

export GaussMarkovChain, chain_from_precision, max_cdf, expected_max,
    argmax_marginals, first_passage, excursion_probability

# ---- normal pdf/cdf (Cephes ndtr, as in the sibling packages) ------

const _SQRTH = 7.07106781186547524401e-1
const _MAXLOG = 7.09782712893383996843e2
const _CT = (9.60497373987051638749e0, 9.00260197203842689217e1,
             2.23200534594684319226e3, 7.00332514112805075473e3,
             5.55923013010394962768e4)
const _CU = (3.35617141647503099647e1, 5.21357949780152679795e2,
             4.59432382970980127987e3, 2.26290000613890934246e4,
             4.92673942608635921086e4)
const _CP = (2.46196981473530512524e-10, 5.64189564831068821977e-1,
             7.46321056442269912687e0, 4.86371970985681366614e1,
             1.96520832956077098242e2, 5.26445194995477358631e2,
             9.34528527171957607540e2, 1.02755188689515710272e3,
             5.57535335369399327526e2)
const _CQ = (1.32281951154744992508e1, 8.67072140885989742329e1,
             3.54937778887819891062e2, 9.75708501743205489753e2,
             1.82390916687909736289e3, 2.24633760818710981792e3,
             1.65666309194161350182e3, 5.57535340817727675546e2)
const _CR = (5.64189583547755073984e-1, 1.27536670759978104416e0,
             5.01905042251180477414e0, 6.16021097993053585195e0,
             7.40974269950448939160e0, 2.97886665372100240670e0)
const _CS = (2.26052863220117276590e0, 9.39603524938001434673e0,
             1.20489539808096656605e1, 1.70814450747565897222e1,
             9.60896809063285878198e0, 3.36907645100081516050e0)

@inline function _polevl(x::Real, C)
    y = zero(x) + C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _p1evl(x::Real, C)
    y = x + C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _erf(x::Real)
    abs(x) > 1.0 && return 1.0 - _erfc(x)
    z = x * x
    return x * _polevl(z, _CT) / _p1evl(z, _CU)
end

@inline function _erfc(a::Real)
    x = abs(a)
    x < 1.0 && return 1.0 - _erf(a)
    z = -a * a
    z < -_MAXLOG && return a < 0 ? 2.0 : 0.0
    z = exp(z)
    p = x < 8.0 ? _polevl(x, _CP) : _polevl(x, _CR)
    q = x < 8.0 ? _p1evl(x, _CQ) : _p1evl(x, _CS)
    y = (z * p) / q
    a < 0 && (y = 2.0 - y)
    return y
end

function ndtr(a::Real)
    isinf(a) && return a > 0 ? 1.0 : 0.0
    x = a * _SQRTH
    z = abs(x)
    z < 1.0 && return 0.5 + 0.5 * _erf(x)
    y = 0.5 * _erfc(z)
    return x > 0 ? 1.0 - y : y
end

npdf(z) = exp(-0.5 * z * z) / sqrt(2 * pi)

# ---- the chain -----------------------------------------------------

"""A Gauss-Markov chain X_1..X_n:
X_1 ~ N(mu[1], sd0^2), X_{t+1} | X_t ~ N(mu[t+1] + phi[t] (X_t - mu[t]),
s[t]^2). `phi` and `s` have length n-1."""
struct GaussMarkovChain
    mu::Vector{Float64}
    sd0::Float64
    phi::Vector{Float64}
    s::Vector{Float64}
end

Base.length(c::GaussMarkovChain) = length(c.mu)

"""Marginal standard deviations, by the variance recursion."""
function marginal_sd(c::GaussMarkovChain)
    n = length(c)
    v = zeros(n)
    v[1] = c.sd0^2
    for t in 1:(n - 1)
        v[t + 1] = c.phi[t]^2 * v[t] + c.s[t]^2
    end
    return sqrt.(v)
end

"""Read the Gauss-Markov chain off a TRIDIAGONAL precision matrix.
The Cholesky of a tridiagonal Q is bidiagonal, so U (X - mu) = z gives
X_t | X_{t+1} ~ N(mu_t - (U[t,t+1]/U[t,t]) (X_{t+1} - mu_{t+1}),
1/U[t,t]^2): conditionals running from the last index to the first.
Factorizing the index-reversed problem therefore returns a chain in
the CALLER's index order, which is what every query reports against.
REFUSES non-tridiagonal precision:
wider bandwidth is a different (lifted) algorithm, and general
sparsity belongs to sampling methods (see Bolin-Lindgren excursions);
neither is approximated silently."""
function chain_from_precision(mu::AbstractVector, Q::AbstractMatrix;
                              tol = 0.0)
    n = length(mu)
    size(Q) == (n, n) || error("Q must be n x n")
    # strict tridiagonality check
    if Q isa SparseMatrixCSC
        I_, J_, V_ = findnz(Q)
        for (i, j, v) in zip(I_, J_, V_)
            abs(i - j) > 1 && abs(v) > tol && error(_refusal_msg())
        end
    else
        for i in 1:n, j in 1:n
            abs(i - j) > 1 && abs(Q[i, j]) > tol && error(_refusal_msg())
        end
    end
    # The bidiagonal Cholesky reads conditionals X_t | X_{t+1}, i.e. a
    # chain running from the LAST index to the first. Reversing both
    # arguments first therefore returns the chain in the CALLER's index
    # order, which is what every downstream query reports against.
    muv = reverse(Float64.(collect(mu)))
    Qv = Matrix(Q)[end:-1:1, end:-1:1]
    F = cholesky(Symmetric(Qv))
    U = F.U                                # upper bidiagonal
    # chain position k is caller index k, so the means pass through
    mur = Float64.(collect(mu))
    phi = zeros(n - 1)
    s = zeros(n - 1)
    for k in 1:(n - 1)
        t = n - k                          # original index of the child
        phi[k] = -U[t, t + 1] / U[t, t]
        s[k] = 1.0 / U[t, t]
    end
    # the reversed chain starts at X_n, whose MARGINAL sd is
    # sqrt((Q^{-1})[n, n]) -- one triangular solve
    en = zeros(n)
    en[n] = 1.0
    sd0 = sqrt((F \ en)[n])
    return GaussMarkovChain(mur, sd0, phi, s)
end

_refusal_msg() = string(
    "precision is not tridiagonal: bandwidth > 1 needs the lifted-",
    "state pass (roadmap; see research/tridiagonal exp4 in the ",
    "winning repo) and general sparse precision belongs to sampling ",
    "methods (Bolin-Lindgren excursions). Refusing rather than ",
    "approximating silently.")

# ---- lattice -------------------------------------------------------

function _grid(c::GaussMarkovChain, points::Int; pad = 8.0)
    msd = marginal_sd(c)
    lo = minimum(c.mu .- pad .* max.(msd, 1e-12))
    hi = maximum(c.mu .+ pad .* max.(msd, 1e-12))
    x = collect(range(lo, hi, length = points))
    return x, x[2] - x[1]
end

"""Precompute the n-1 transition matrices T[t][i, j] =
P(X_{t+1} = x_i | X_t = x_j) dx on the grid: O(n L^2) memory, and
every restricted pass thereafter is BLAS slices."""
function _transitions(c::GaussMarkovChain, x::Vector{Float64},
                      dx::Float64)
    n = length(c)
    L = length(x)
    T = Vector{Matrix{Float64}}(undef, n - 1)
    for t in 1:(n - 1)
        M = Matrix{Float64}(undef, L, L)
        for (j, xm) in enumerate(x)
            m = c.mu[t + 1] + c.phi[t] * (xm - c.mu[t])
            invs = 1.0 / c.s[t]
            @inbounds for i in 1:L
                M[i, j] = npdf((x[i] - m) * invs) * invs * dx
            end
        end
        T[t] = M
    end
    return T
end

_initial_density(c, x, dx) =
    [npdf((xi - c.mu[1]) / c.sd0) / c.sd0 * dx for xi in x]

# ---- max CDF and first passage (one restricted forward pass) -------

"""P(max_t X_t <= u). `u` may be a number or a vector."""
function max_cdf(c::GaussMarkovChain, u::Real; points = 400)
    x, dx = _grid(c, points)
    T = _transitions(c, x, dx)
    return _restricted_masses(c, T, x, dx, float(u))[end]
end

function max_cdf(c::GaussMarkovChain, us::AbstractVector; points = 400)
    x, dx = _grid(c, points)
    T = _transitions(c, x, dx)
    return [_restricted_masses(c, T, x, dx, float(u))[end] for u in us]
end

function _restricted_masses(c, T, x, dx, u)
    n = length(c)
    # fractional occupancy of the boundary cell: v[i] is mass in
    # [x_i - dx/2, x_i + dx/2), and a hard cutoff at u costs O(dx);
    # keeping the sub-cell fraction restores O(dx^2)
    keep = clamp.((u .- (x .- dx / 2)) ./ dx, 0.0, 1.0)
    v = _initial_density(c, x, dx) .* keep
    masses = zeros(n)
    masses[1] = sum(v)
    for t in 1:(n - 1)
        v = keep .* (T[t] * v)
        masses[t + 1] = sum(v)
    end
    return masses
end

"""P(max <= u) complement: P(any X_t > u) -- the excursion
probability of Bolin-Lindgren, exact on the chain."""
excursion_probability(c::GaussMarkovChain, u; points = 400) =
    1 .- max_cdf(c, u; points = points)

"""Distribution of the FIRST index exceeding u: a vector p with
p[t] = P(first passage at t), plus P(never) as the final entry."""
function first_passage(c::GaussMarkovChain, u::Real; points = 400)
    x, dx = _grid(c, points)
    T = _transitions(c, x, dx)
    masses = _restricted_masses(c, T, x, dx, float(u))
    n = length(c)
    p = zeros(n + 1)
    prev = 1.0
    for t in 1:n
        p[t] = prev - masses[t]
        prev = masses[t]
    end
    p[n + 1] = masses[n]
    return p
end

"""E[max_t X_t], by integrating the max survival function on a
threshold grid spanning the chain's range."""
function expected_max(c::GaussMarkovChain; points = 400, nu = 200)
    msd = marginal_sd(c)
    lo = minimum(c.mu) - 8.0 * maximum(msd)
    hi = maximum(c.mu) + 8.0 * maximum(msd)
    us = collect(range(lo, hi, length = nu))
    du = us[2] - us[1]
    x, dx = _grid(c, points)
    T = _transitions(c, x, dx)
    F = [_restricted_masses(c, T, x, dx, u)[end] for u in us]
    # trapezoid in the threshold: the left-rectangle rule biased
    # E[max] by O(du) (measured +0.053 at nu = 200 against MC)
    surv = 1.0 .- F
    return lo + (sum(surv) - surv[1] / 2 - surv[end] / 2) * du
end

# ---- argmax marginals (per-threshold forward and backward) ---------

"""P(X_t is the maximum), for every t: the layer the Kalman smoother
does not have. For each grid threshold u_k, a forward pass restricted
strictly below u_k reaches X_t = u_k (alpha), and a backward
conditional pass keeps the future below u_k (beta); summing
alpha_t(k) beta_t(k) over k integrates out the value of the maximum.
BLAS slices of precomputed transitions; ties are measure-zero in the
continuum and unresolved on the grid beyond its resolution."""
function argmax_marginals(c::GaussMarkovChain; points = 300)
    n = length(c)
    x, dx = _grid(c, points)
    T = _transitions(c, x, dx)
    p0 = _initial_density(c, x, dx)
    P = zeros(n)
    for k in 2:points
        lo = 1:k
        # "strictly below u_k" keeps cells 1..k-1 plus HALF of the
        # boundary cell k (values in [x_k - dx/2, x_k)): the same
        # fractional-occupancy correction as the restricted passes
        w = ones(k)
        w[k] = 0.5
        # backward: betas[t] = P(X_{t+1}..X_n < u_k | X_t = u_k)
        betas = zeros(n)
        betas[n] = 1.0
        b = ones(k)
        for t in (n - 1):-1:1
            full = (@view T[t][lo, :])' * (w .* b)   # b_t on the grid
            betas[t] = full[k]
            b = full[lo]
        end
        # forward: alpha_t = P(X_1..X_{t-1} < u_k, X_t = u_k)
        v = p0[lo]
        P[1] += p0[k] * betas[1]
        for t in 1:(n - 1)
            q = (@view T[t][lo, lo]) * (w .* v)
            P[t + 1] += q[k] * betas[t + 1]
            v = q
        end
    end
    total = sum(P)
    (isfinite(total) && abs(total - 1) <= 0.05) ||
        error("argmax marginals defective (mass $total): raise points=")
    return P ./ total
end

end # module
