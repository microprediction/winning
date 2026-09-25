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

using LinearAlgebra: SymTridiagonal
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
Non-tridiagonal precision is refused:
wider bandwidth is a different (lifted) algorithm, and general
sparsity belongs to sampling methods (see Bolin-Lindgren excursions);
neither is approximated silently. A `SymTridiagonal` is accepted
directly and costs O(n) time and memory; a dense or sparse matrix is
checked for bandwidth (its upper triangle is used) and then routed the
same way, so a GMRF's sparse precision is never densified."""
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
    dv = [Float64(Q[i, i]) for i in 1:n]
    ev = [Float64(Q[i, i + 1]) for i in 1:(n - 1)]
    return chain_from_precision(mu, SymTridiagonal(dv, ev))
end

function chain_from_precision(mu::AbstractVector, Q::SymTridiagonal)
    n = length(mu)
    size(Q, 1) == n || error("Q must be n x n")
    n >= 1 || error("empty chain")
    # The bidiagonal Cholesky R' R = Q reads conditionals X_t | X_{t+1},
    # a chain running from the LAST index to the first. Factorizing the
    # index-reversed matrix therefore returns the chain in the CALLER's
    # index order, which is what every downstream query reports against.
    dr = reverse(Float64.(collect(Q.dv)))
    er = reverse(Float64.(collect(Q.ev)))
    r = zeros(n)                       # diagonal of R
    sup = zeros(max(n - 1, 0))         # superdiagonal of R
    dr[1] > 0 || error("precision is not positive definite")
    r[1] = sqrt(dr[1])
    for t in 1:(n - 1)
        sup[t] = er[t] / r[t]
        v = dr[t + 1] - sup[t]^2
        v > 0 || error("precision is not positive definite")
        r[t + 1] = sqrt(v)
    end
    # chain position k is caller index k, so the means pass through
    mur = Float64.(collect(mu))
    phi = zeros(max(n - 1, 0))
    s = zeros(max(n - 1, 0))
    for k in 1:(n - 1)
        t = n - k                      # reversed index of the child
        phi[k] = -sup[t] / r[t]
        s[k] = 1.0 / r[t]
    end
    # the reversed chain starts at X_n, whose marginal variance is the
    # last Schur complement, (Qv^{-1})[n, n] = 1 / R[n, n]^2
    sd0 = 1.0 / r[n]
    return GaussMarkovChain(mur, sd0, phi, s)
end

_refusal_msg() = string(
    "precision is not tridiagonal: bandwidth > 1 needs the lifted-",
    "state pass (roadmap; see research/tridiagonal exp4 in the ",
    "winning repo) and general sparse precision belongs to sampling ",
    "methods (Bolin-Lindgren excursions). Refusing rather than ",
    "approximating silently.")

# ---- lattice -------------------------------------------------------

"""Largest lattice a chain of length `n` may use.

The transitions are (n-1) dense L x L matrices, so the memory is
O(n L^2). This budget is 2e7 doubles, about 160 MB, which is what keeps
a threshold far out in the tail from asking for a lattice nobody can
hold.
"""
_max_points(n::Int) = n <= 1 ? 2_000_001 :
    max(400, floor(Int, sqrt(2.0e7 / (n - 1))))

"""
    _grid(c, points; pad = 8.0, u = nothing)

The lattice, extended to cover a threshold `u` that lies outside it.

The grid was built from the chain's means and marginal sds alone, so a
threshold above the last cell left the occupancy identically one and
`max_cdf`, `excursion_probability` and `first_passage` stopped depending
on `u` at all: 9, 10 and 20 sd out all returned the same 1.11e-15, which
is quadrature noise rather than a tail (#197).

Extending the range without adding points would coarsen the bulk, so the
spacing is preserved and the point count grows with the range, up to
`_max_points`. Past that the spacing does grow, and the caller is told.
"""
function _grid(c::GaussMarkovChain, points::Int; pad = 8.0, u = nothing)
    msd = marginal_sd(c)
    lo = minimum(c.mu .- pad .* max.(msd, 1e-12))
    hi = maximum(c.mu .+ pad .* max.(msd, 1e-12))
    if u !== nothing
        uf = float(u)
        if isfinite(uf)
            margin = pad * maximum(max.(msd, 1e-12))
            span0 = hi - lo
            lo = min(lo, uf - margin)
            hi = max(hi, uf + margin)
            if hi - lo > span0
                want = ceil(Int, (points - 1) * (hi - lo) / span0) + 1
                cap = _max_points(length(c))
                if want > cap
                    @warn string("threshold ", uf, " is far outside the ",
                                 "chain's range; the lattice is capped at ",
                                 cap, " points, so the spacing is coarser ",
                                 "than requested")
                end
                points = min(want, cap)
            end
        end
    end
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
            @inbounds for i in 1:L
                M[i, j] = _cell_mass(x[i], m, c.s[t], dx)
            end
        end
        T[t] = M
    end
    return T
end

"""
    _cell_mass(xi, m, invs, dx)

The probability that a N(m, 1/invs^2) draw lands in the lattice cell
centred on `xi`, as a CDF difference rather than density x spacing.

`npdf(z) * invs * dx` is a Riemann sample of the density, and it is only
a probability when the density is flat across the cell. An innovation sd
far below the spacing makes the transition kernel a SPIKE between
samples: with innovation variance 1e-4 on a lattice whose spacing is set
by the marginal sd, each column of the transition matrix summed to about
80 instead of 1, and `max_cdf` returned 79.988 for a probability (#244).
An excursion probability came back as -78.99.

The CDF difference is the cell's exact mass, so a column sums to one
however narrow the kernel is, and for a wide kernel it agrees with the
old expression to O(dx^2) -- it is strictly the better quadrature, not a
special case. Renormalising the columns instead would have hidden the
resolution failure rather than fixed it.
"""
@inline function _cell_mass(xi::Float64, m::Float64, sd::Float64,
                            dx::Float64)
    # Integrating the kernel over the cell is exact in mass but convolves
    # a boxcar of width dx at every step, which adds dx^2/12 to the
    # transition variance. Deflating by exactly that much puts it back,
    # so the discretised chain has the right variance and this is not a
    # worse quadrature than the midpoint rule where the midpoint rule
    # works. Below the resolution limit the deflated sd is zero and the
    # cell containing the mean takes the whole mass -- the correct
    # degenerate limit, and the case the midpoint rule turned into 80.
    v = sd * sd - dx * dx / 12.0
    h = 0.5 * dx
    if v <= 0.0
        return (m >= xi - h && m < xi + h) ? 1.0 : 0.0
    end
    inv = 1.0 / sqrt(v)
    mass = ndtr((xi + h - m) * inv) - ndtr((xi - h - m) * inv)
    mass > 0.0 && return mass
    # Both endpoints sit in the same saturated tail of ndtr, which
    # reaches exactly 1 around 8.3 sd, so their difference underflows to
    # zero and a threshold 9 sd out returned a tail probability of
    # exactly 0 for every threshold alike (#197). The midpoint density is
    # the form that survives out there -- it does not underflow until
    # about 38 sd -- and where both are representable the two agree to
    # O(dx^2), so this is a fallback, not a second opinion.
    return npdf((xi - m) * inv) * inv * dx
end

_initial_density(c, x, dx) =
    [_cell_mass(xi, c.mu[1], c.sd0, dx) for xi in x]

# ---- max CDF and first passage (one restricted forward pass) -------

"""P(max_t X_t <= u). `u` may be a number or a vector."""
function max_cdf(c::GaussMarkovChain, u::Real; points = 400)
    x, dx = _grid(c, points; u = u)
    T = _transitions(c, x, dx)
    return _restricted_masses(c, T, x, dx, float(u))[1][end]
end

function max_cdf(c::GaussMarkovChain, us::AbstractVector; points = 400)
    x, dx = _grid(c, points; u = isempty(us) ? nothing : maximum(float.(us)))
    T = _transitions(c, x, dx)
    return [_restricted_masses(c, T, x, dx, float(u))[1][end] for u in us]
end

"""Restricted masses, and the mass that EXCEEDS u at each step.

`masses[t]` is P(X_1..X_t all <= u). The exceedance used to be taken as
`1 - masses[n]`, and in the tail that is a subtraction of two numbers
that agree to every bit: at 10 sd it returned -8.9e-16, a negative
probability, and `first_passage` differenced the same quantities and
produced a negative passage mass.

`escaped[t]` is the mass removed at step t, summed over the cells ABOVE
u rather than subtracted from one. Those are small positive numbers, so
the answer has full relative accuracy however far out the threshold is.
"""
function _restricted_masses(c, T, x, dx, u)
    n = length(c)
    # fractional occupancy of the boundary cell: v[i] is mass in
    # [x_i - dx/2, x_i + dx/2), and a hard cutoff at u costs O(dx);
    # keeping the sub-cell fraction restores O(dx^2)
    keep = clamp.((u .- (x .- dx / 2)) ./ dx, 0.0, 1.0)
    drop = 1.0 .- keep
    masses = zeros(n)
    escaped = zeros(n)
    w = _initial_density(c, x, dx)
    escaped[1] = sum(w .* drop)
    v = w .* keep
    masses[1] = sum(v)
    for t in 1:(n - 1)
        w = T[t] * v
        escaped[t + 1] = sum(w .* drop)
        v = keep .* w
        masses[t + 1] = sum(v)
    end
    return masses, escaped
end

"""P(max <= u) complement: P(any X_t > u) -- the excursion
probability of Bolin-Lindgren, exact on the chain."""
function excursion_probability(c::GaussMarkovChain, u::Real; points = 400)
    x, dx = _grid(c, points; u = u)
    T = _transitions(c, x, dx)
    _masses, escaped = _restricted_masses(c, T, x, dx, float(u))
    return sum(escaped)          # small positive terms, not 1 - (1 - eps)
end

excursion_probability(c::GaussMarkovChain, us::AbstractVector; points = 400) =
    [excursion_probability(c, u; points = points) for u in us]

"""Distribution of the FIRST index exceeding u: a vector p with
p[t] = P(first passage at t), plus P(never) as the final entry."""
function first_passage(c::GaussMarkovChain, u::Real; points = 400)
    x, dx = _grid(c, points; u = u)
    T = _transitions(c, x, dx)
    masses, escaped = _restricted_masses(c, T, x, dx, float(u))
    n = length(c)
    p = zeros(n + 1)
    # the mass removed AT step t is the first-passage mass at t, taken
    # directly rather than as a difference of two near-one numbers
    for t in 1:n
        p[t] = escaped[t]
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
    F = [_restricted_masses(c, T, x, dx, u)[1][end] for u in us]
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
