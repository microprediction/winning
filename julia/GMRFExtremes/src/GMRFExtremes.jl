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

using LinearAlgebra: SymTridiagonal, mul!
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
s[t]^2). `phi` and `s` have length n-1.

The constructor enforces that shape. It used to be a bare field struct,
so surplus `phi`/`s` entries were accepted and silently ignored by every
query (they all loop over `length(mu)`), and short ones failed later with
a raw BoundsError (#347). Means and coefficients must be finite and the
standard deviations finite and nonnegative."""
struct GaussMarkovChain
    mu::Vector{Float64}
    sd0::Float64
    phi::Vector{Float64}
    s::Vector{Float64}
    function GaussMarkovChain(mu::AbstractVector{<:Real}, sd0::Real,
                              phi::AbstractVector{<:Real},
                              s::AbstractVector{<:Real})
        n = length(mu)
        n >= 1 || throw(ArgumentError("a chain needs at least one node"))
        (length(phi) == n - 1 && length(s) == n - 1) || throw(
            DimensionMismatch(string(
                "phi and s must have length n - 1 = ", n - 1, " for ", n,
                " means; got length(phi) = ", length(phi),
                ", length(s) = ", length(s))))
        all(isfinite, mu) || throw(ArgumentError("mu must be finite"))
        all(isfinite, phi) || throw(ArgumentError("phi must be finite"))
        (isfinite(sd0) && sd0 >= 0) ||
            throw(ArgumentError("sd0 must be finite and nonnegative"))
        all(v -> isfinite(v) && v >= 0, s) ||
            throw(ArgumentError("s must be finite and nonnegative"))
        return new(Float64.(collect(mu)), Float64(sd0),
                   Float64.(collect(phi)), Float64.(collect(s)))
    end
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
checked for bandwidth, finiteness and symmetry (to `tol`) and then
routed the same way, so a GMRF's sparse precision is never densified."""
function chain_from_precision(mu::AbstractVector, Q::AbstractMatrix;
                              tol = 0.0)
    n = length(mu)
    size(Q) == (n, n) || error("Q must be n x n")
    # tol suppresses numerical residue; it must not be able to switch the
    # refusal off. tol = Inf passed every bandwidth and symmetry check and
    # the off-band entries were dropped: a distance-two coupling priced
    # 0.125 for a true 0.1049 (#562). NaN and negative values had
    # accidental comparison semantics.
    (tol isa Real && isfinite(tol) && tol >= 0) || throw(ArgumentError(
        "tol must be a finite, nonnegative number; got " * string(tol)))
    tol = Float64(tol)
    # strict tridiagonality check
    if Q isa SparseMatrixCSC
        I_, J_, V_ = findnz(Q)
        for (i, j, v) in zip(I_, J_, V_)
            isfinite(v) || error("precision has a non-finite entry")
            abs(i - j) > 1 && abs(v) > tol && error(_refusal_msg())
        end
    else
        for i in 1:n, j in 1:n
            isfinite(Q[i, j]) || error("precision has a non-finite entry")
            abs(i - j) > 1 && abs(Q[i, j]) > tol && error(_refusal_msg())
        end
    end
    # A precision is symmetric. Only the upper off-diagonal is read below,
    # so an asymmetric matrix used to be replaced in silence by the
    # symmetric completion of its upper triangle: a lower-only
    # off-diagonal priced as a diagonal Q, and [2 -1; -0.1 2] as
    # [2 -1; -1 2] (#356). `tol` is the absolute tolerance, plus a few
    # ulps of the entries so a numerically assembled Q still passes.
    for i in 1:(n - 1)
        a, b = Float64(Q[i, i + 1]), Float64(Q[i + 1, i])
        abs(a - b) <= tol + 8 * eps(max(abs(a), abs(b))) || error(string(
            "precision is not symmetric: Q[", i, ",", i + 1, "] = ", a,
            " but Q[", i + 1, ",", i, "] = ", b, ". Refusing rather than ",
            "reading only the upper triangle"))
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

"""Largest lattice a chain of length `n` may use when a threshold
widens the grid.

The work is O(n L^2) transition entries, so the point count is scaled
down with the chain length, with a floor of 400. Memory is NOT O(n L^2):
the passes stream one L x L transition at a time (they used to keep all
n - 1 of them, 1.28 GB at n = 1000 under a promised 160 MB, #533), and
`_LATTICE_DOUBLES` bounds that one matrix.
"""
_max_points(n::Int) = n <= 1 ? 2_000_001 :
    max(400, floor(Int, sqrt(2.0e7 / (n - 1))))

"""Memory budget, in doubles, for one dense L x L lattice matrix
(about 160 MB): `points` above `isqrt(_LATTICE_DOUBLES)` = 4472 on a
chain with transitions is refused before anything is allocated."""
const _LATTICE_DOUBLES = 20_000_000

function _check_lattice(L::Int, n::Int)
    n >= 2 || return nothing             # no transition matrix at all
    L * L <= _LATTICE_DOUBLES || throw(ArgumentError(string(
        "a ", L, "-point lattice needs ", L, " x ", L, " transition ",
        "matrices (", round(8 * L^2 / 1e6; digits = 1), " MB each), over ",
        "the ", 8 * _LATTICE_DOUBLES ÷ 1_000_000, " MB budget; use at most ",
        isqrt(_LATTICE_DOUBLES), " points")))
    return nothing
end

"""A lattice point budget must be a whole number of at least 2: one
point has no spacing (#550)."""
function _check_points(points)
    (points isa Integer && points >= 2) || throw(ArgumentError(
        "points must be an integer >= 2 (a lattice needs a spacing); " *
        "got " * string(points)))
    return Int(points)
end

"""Thresholds are ordered boundaries: NaN is not one. It was dropped
from grid sizing and then made every cell's occupancy NaN, so a one-node
chain returned a zero-mass law (max_cdf and its complement both 0,
first_passage summing to 0) and longer chains returned NaN (#513).
Infinite thresholds are the exact limits and are kept."""
function _check_threshold(u)
    if u isa Real
        isnan(u) && throw(ArgumentError(
            "threshold u is NaN; a threshold must be a number or +-Inf"))
    else
        for (i, v) in enumerate(u)
            (v isa Real && !isnan(v)) || throw(ArgumentError(
                "threshold u[" * string(i) * "] = " * string(v) *
                " is not a number; a threshold must be real or +-Inf"))
        end
    end
    return nothing
end

"""
    _grid(c, points; pad = 8.0, u = nothing, align = false)

The lattice, extended to cover the threshold(s) `u` (a number or a
collection) that lie outside it.

The grid was built from the chain's means and marginal sds alone, so a
threshold above the last cell left the occupancy identically one and
`max_cdf`, `excursion_probability` and `first_passage` stopped depending
on `u` at all: 9, 10 and 20 sd out all returned the same 1.11e-15, which
is quadrature noise rather than a tail (#197). A vector of thresholds is
covered at BOTH ends: the vector overload used to pass only
`maximum(us)`, so a lower-tail entry fell off the grid and came back as
exactly zero while the scalar call answered it (#417).

Extending the range without adding points would coarsen the bulk, so the
spacing is preserved and the point count grows with the range, up to
`_max_points`. Past that the spacing does grow, and the caller is told.

The scale is the chain's own: the sds used to be floored at an ABSOLUTE
1e-12, so a chain in units below that was gridded on a range unrelated to
its spread and `max_cdf(a X, a u)` stopped equalling `max_cdf(X, u)`
(0.625 against 0.841 at a = 1e-14, #363). The floor is now relative to
the largest marginal sd; the absolute one survives only for a chain with
no randomness at all.

`align = true` (scalar `u` only) slides the lattice by under half a cell
so that `u` is a cell EDGE. The restriction to `<= u` is then exact on
the lattice rather than a uniform-within-cell guess, which matters when
the guess is wrong: an exactly repeated variable re-clipped the same
boundary cell at every step and invented first-passage mass after step 1
(#244).
"""
function _grid(c::GaussMarkovChain, points; pad = 8.0, u = nothing,
               align = false)
    points = _check_points(points)
    msd = marginal_sd(c)
    big = maximum(msd)
    floor_ = big > 0 ? 1e-12 * big : 1e-12
    sds = max.(msd, floor_)
    lo = minimum(c.mu .- pad .* sds)
    hi = maximum(c.mu .+ pad .* sds)
    us = u === nothing ? Float64[] :
         filter(isfinite, u isa Real ? [float(u)] : Float64[float(v) for v in u])
    if !isempty(us)
        margin = pad * maximum(sds)
        span0 = hi - lo
        lo = min(lo, minimum(us) - margin)
        hi = max(hi, maximum(us) + margin)
        if hi - lo > span0
            want = ceil(Int, (points - 1) * (hi - lo) / span0) + 1
            cap = _max_points(length(c))
            if want > cap
                @warn string("threshold(s) ", extrema(us), " far outside ",
                             "the chain's range; the lattice is capped at ",
                             cap, " points, so the spacing is coarser ",
                             "than requested")
            end
            points = min(want, cap)
        end
    end
    dx = (hi - lo) / (points - 1)
    if align && length(us) == 1
        # put u on a cell edge: edges sit at lo - dx/2 + k dx
        f = (us[1] - (lo - dx / 2)) / dx
        shift = (f - round(f)) * dx            # |shift| <= dx/2
        lo += shift
        hi += shift
    end
    _check_lattice(points, length(c))
    x = collect(range(lo, hi, length = points))
    return x, x[2] - x[1]
end

"""Mass of N(m, sd^2) on [a, b), tail-stable.

The difference is taken in whichever tail the interval sits, as
`ndtr(-za) - ndtr(-zb)` above the mean, so it does not saturate where
`ndtr` reaches exactly one (about 8.3 sd); the midpoint density is the
last resort only where even that underflows (#197). `sd == 0` is the
point mass at m."""
@inline function _interval_mass(a::Float64, b::Float64, m::Float64,
                                sd::Float64)
    b > a || return 0.0
    if sd <= 0.0
        return (m >= a && m < b) ? 1.0 : 0.0
    end
    za = (a - m) / sd
    zb = (b - m) / sd
    mass = za >= 0.0 ? ndtr(-za) - ndtr(-zb) : ndtr(zb) - ndtr(za)
    mass > 0.0 && return mass
    isfinite(za) && isfinite(zb) || return 0.0
    return npdf(0.5 * (za + zb)) * (zb - za)
end

"""The transition matrix M[i, j] = P(X_{t+1} in cell i | X_t = x_j) on
the grid, written into `M` (L x L). The passes build one at a time and
stream it, so memory is O(L^2) whatever the chain length (#533).

Each entry is a cell PROBABILITY, a CDF difference, never density x
spacing. `npdf(z) * dx / s` is a Riemann sample and only a probability
where the density is flat across the cell: with innovation variance 1e-4
on a lattice spaced by the marginal sd each column summed to about 80
and `max_cdf` returned 79.988 (#244).

The lattice carries each cell's mass at its centre, which drops the
within-cell spread of the SOURCE, phi^2 dx^2/12 after the map, while
binning the target is exact. So the kernel variance is deflated by
phi^2 dx^2/12 (it used to be dx^2/12 whatever phi was, which is right
only for a random walk). Where that leaves nothing -- a kernel narrower
than the lattice -- the conditional mean is split linearly between the
two cells around it rather than snapped into one, which keeps a
sub-cell drift instead of discarding it at every step."""
function _transition!(M::Matrix{Float64}, c::GaussMarkovChain,
                      x::Vector{Float64}, dx::Float64, t::Int)
    L = length(x)
    h = 0.5 * dx
    fill!(M, 0.0)
    begin
        ph = c.phi[t]
        v = c.s[t]^2 - ph * ph * dx * dx / 12.0
        sdv = v > 0.0 ? sqrt(v) : 0.0
        for (j, xm) in enumerate(x)
            m = c.mu[t + 1] + ph * (xm - c.mu[t])
            if sdv > 0.0
                @inbounds for i in 1:L
                    M[i, j] = _interval_mass(x[i] - h, x[i] + h, m, sdv)
                end
            else
                q = (m - x[1]) / dx + 1.0        # fractional cell index
                i0 = floor(Int, q)
                f = q - i0
                1 <= i0 <= L && (M[i0, j] += 1.0 - f)
                1 <= i0 + 1 <= L && (M[i0 + 1, j] += f)
            end
        end
    end
    return M
end

"""All n-1 transition matrices (O(n L^2) memory; tests and research
only -- the queries stream `_transition!`)."""
_transitions(c::GaussMarkovChain, x::Vector{Float64}, dx::Float64) =
    [_transition!(zeros(length(x), length(x)), c, x, dx, t)
     for t in 1:(length(c) - 1)]

"""Exact cell probabilities of X_1. No variance deflation: there is no
source cell whose spread was dropped, and deflating made a sub-cell
`sd0` a point mass that a cell-edge threshold then discarded whole
(`max_cdf` 5.5e-15 for a true 0.25, #244)."""
_initial_density(c, x, dx) =
    [_interval_mass(xi - dx / 2, xi + dx / 2, c.mu[1], c.sd0) for xi in x]

# ---- max CDF and first passage (one restricted forward pass) -------

"""P(max_t X_t <= u). `u` may be a number or a vector.

A vector shares one lattice, which covers every threshold in it, so the
batch agrees with the scalar calls to quadrature accuracy (#417)."""
function max_cdf(c::GaussMarkovChain, u::Real; points = 400)
    _check_threshold(u)
    x, dx = _grid(c, points; u = u, align = true)
    return _restricted_masses(c, x, dx, [float(u)])[1][end, 1]
end

function max_cdf(c::GaussMarkovChain, us::AbstractVector; points = 400)
    _check_threshold(us)
    isempty(us) && return Float64[]
    x, dx = _grid(c, points; u = us)
    masses, _ = _restricted_masses(c, x, dx, Float64[float(u) for u in us])
    return masses[end, :]
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

Step 1 is restricted EXACTLY, by splitting X_1's cell masses at u, so a
sub-cell `sd0` is not assumed uniform across its cell. Later steps keep
the fractional occupancy of the boundary cell, which is exact (0 or 1)
when the caller aligned the lattice to u.

All thresholds `us` share one forward pass, with column k of the state
restricted at `us[k]`; each transition is built once and streamed, so
memory is O(L^2 + L K), not the O(n L^2) of keeping every transition
(#533). Returns (n x K) matrices."""
function _restricted_masses(c, x, dx, us::Vector{Float64})
    n = length(c)
    L = length(x)
    K = length(us)
    h = dx / 2
    # fractional occupancy of the boundary cell: v[i] is mass in
    # [x_i - dx/2, x_i + dx/2), and a hard cutoff at u costs O(dx);
    # keeping the sub-cell fraction restores O(dx^2). On an aligned
    # lattice the fraction is 0 or 1 up to rounding: snap it.
    keep = clamp.((us' .- (x .- h)) ./ dx, 0.0, 1.0)      # (L, K)
    for i in eachindex(keep)
        abs(keep[i] - round(keep[i])) < 1e-9 && (keep[i] = round(keep[i]))
    end
    drop = 1.0 .- keep
    masses = zeros(n, K)
    escaped = zeros(n, K)
    m1, s1 = c.mu[1], c.sd0
    V = [_interval_mass(x[i] - h, min(x[i] + h, us[k]), m1, s1)
         for i in 1:L, k in 1:K]
    for k in 1:K
        escaped[1, k] = sum(_interval_mass(max(xi - h, us[k]), xi + h, m1, s1)
                            for xi in x)
        masses[1, k] = sum(@view V[:, k])
    end
    n >= 2 || return masses, escaped
    M = Matrix{Float64}(undef, L, L)
    W = Matrix{Float64}(undef, L, K)
    for t in 1:(n - 1)
        _transition!(M, c, x, dx, t)
        mul!(W, M, V)
        for k in 1:K
            e = 0.0
            m = 0.0
            @inbounds for i in 1:L
                e += W[i, k] * drop[i, k]
                vi = keep[i, k] * W[i, k]
                V[i, k] = vi
                m += vi
            end
            escaped[t + 1, k] = e
            masses[t + 1, k] = m
        end
    end
    return masses, escaped
end

"""P(max <= u) complement: P(any X_t > u) -- the excursion
probability of Bolin-Lindgren, exact on the chain."""
function excursion_probability(c::GaussMarkovChain, u::Real; points = 400)
    _check_threshold(u)
    x, dx = _grid(c, points; u = u, align = true)
    _masses, escaped = _restricted_masses(c, x, dx, [float(u)])
    return sum(escaped)          # small positive terms, not 1 - (1 - eps)
end

function excursion_probability(c::GaussMarkovChain, us::AbstractVector;
                               points = 400)
    _check_threshold(us)                 # every entry before any work
    return [excursion_probability(c, u; points = points) for u in us]
end

"""Distribution of the FIRST index exceeding u: a vector p with
p[t] = P(first passage at t), plus P(never) as the final entry."""
function first_passage(c::GaussMarkovChain, u::Real; points = 400)
    _check_threshold(u)
    x, dx = _grid(c, points; u = u, align = true)
    masses, escaped = _restricted_masses(c, x, dx, [float(u)])
    n = length(c)
    p = zeros(n + 1)
    # the mass removed AT step t is the first-passage mass at t, taken
    # directly rather than as a difference of two near-one numbers
    for t in 1:n
        p[t] = escaped[t, 1]
    end
    p[n + 1] = masses[n, 1]
    return p
end

"""E[max_t X_t], by integrating the max survival function on a
threshold grid spanning the chain's range."""
function expected_max(c::GaussMarkovChain; points = 400, nu = 200)
    # the trapezoid needs two thresholds; nu = 0 or 1 read us[2] (#550)
    (nu isa Integer && nu >= 2) || throw(ArgumentError(
        "nu must be an integer >= 2 (the threshold trapezoid needs two " *
        "points); got " * string(nu)))
    msd = marginal_sd(c)
    lo = minimum(c.mu) - 8.0 * maximum(msd)
    hi = maximum(c.mu) + 8.0 * maximum(msd)
    us = collect(range(lo, hi, length = nu))
    du = us[2] - us[1]
    x, dx = _grid(c, points)
    F = _restricted_masses(c, x, dx, us)[1][end, :]
    # trapezoid in the threshold: the left-rectangle rule biased
    # E[max] by O(du) (measured +0.053 at nu = 200 against MC)
    surv = 1.0 .- F
    return lo + (sum(surv) - surv[1] / 2 - surv[end] / 2) * du
end

# ---- argmax marginals (per-threshold forward and backward) ---------

"""Probability the argmax pass may misassign between adjacent nodes.

When X_t and X_{t+1} land in the same lattice cell the pass cannot tell
which is larger and splits the cell half and half. That is right when
their difference D is symmetric about zero on the scale of a cell and
wrong otherwise: with D = 0.001 + 0.001 Z on a 0.054 lattice, every
draw ties, the split returns [0.5, 0.5], and the truth is
[Phi(-1), Phi(1)] = [0.159, 0.841] (#432). The bias of the split is
|P(0 < D < dx) - P(-dx < D < 0)| / 2 for D ~ N(mu[t+1] - mu[t],
(phi - 1)^2 var_t + s_t^2), the marginal law of the adjacent
difference. Returns the worst (bias, t). It is a conservative heuristic
(measured on two-node chains the realised error is about a sixth of it)
and covers adjacent pairs only: nonadjacent pairs are separated by at
least one further innovation."""
function _tie_bias(c::GaussMarkovChain, dx::Float64)
    n = length(c)
    n <= 1 && return 0.0, 0
    msd = marginal_sd(c)
    worst, at = 0.0, 0
    for t in 1:(n - 1)
        md = c.mu[t + 1] - c.mu[t]
        sdd = sqrt((c.phi[t] - 1)^2 * msd[t]^2 + c.s[t]^2)
        up = _interval_mass(0.0, dx, md, sdd)
        dn = _interval_mass(-dx, 0.0, md, sdd)
        b = abs(up - dn) / 2
        b > worst && ((worst, at) = (b, t))
    end
    return worst, at
end

"""Default refusal threshold for `argmax_marginals`' tie bias."""
const ARGMAX_RESOLUTION_TOL = 1e-2

"""P(X_t is the maximum), for every t: the layer the Kalman smoother
does not have. For each grid threshold u_k, a forward pass restricted
strictly below u_k reaches X_t = u_k (alpha), and a backward
conditional pass keeps the future below u_k (beta); summing
alpha_t(k) beta_t(k) over k integrates out the value of the maximum.
BLAS slices of precomputed transitions; ties are measure-zero in the
continuum and unresolved on the grid beyond its resolution."""
function argmax_marginals(c::GaussMarkovChain; points = 300,
                          resolution_tol = ARGMAX_RESOLUTION_TOL)
    n = length(c)
    x, dx = _grid(c, points)
    err, t = _tie_bias(c, dx)
    err <= resolution_tol || error(string(
        "the lattice cannot resolve the order of X_", t, " and X_", t + 1,
        ": their difference has a sub-cell spread or drift relative to ",
        "the spacing dx = ", dx, ", so the half-cell tie rule would ",
        "misassign up to ", round(err; sigdigits = 3), " of probability ",
        "(tolerance ", resolution_tol, "). Raise points=, or treat the ",
        "pair analytically; refusing rather than returning a 50/50 ",
        "split (#432)"))
    L = length(x)
    p0 = _initial_density(c, x, dx)
    P = zeros(n)
    # Every threshold u_k = x_k (k = 2..L) at once: column k of the
    # state holds the pass restricted below u_k, so its rows above k are
    # zero (upper-triangular). "Strictly below u_k" keeps cells 1..k-1
    # plus HALF of the boundary cell k (values in [x_k - dx/2, x_k)):
    # the same fractional-occupancy correction as the restricted passes.
    # Each transition is built when needed and streamed -- once for the
    # backward pass, once for the forward -- instead of keeping all n-1
    # (O(n L^2) memory, #533); only the n x L betas are kept.
    Wt = [i < k ? 1.0 : i == k ? 0.5 : 0.0 for i in 1:L, k in 1:L]
    Wt[:, 1] .= 0.0                      # u_1 is not a threshold
    mask = [i <= k ? 1.0 : 0.0 for i in 1:L, k in 1:L]
    M = Matrix{Float64}(undef, L, L)
    A = Matrix{Float64}(undef, L, L)
    B = similar(A)
    # backward: betas[t, k] = P(X_{t+1}..X_n < u_k | X_t = u_k)
    betas = zeros(n, L)
    betas[n, :] .= 1.0
    S = copy(mask)                       # b for every k, zero above k
    for t in (n - 1):-1:1
        _transition!(M, c, x, dx, t)
        A .= Wt .* S
        mul!(B, M', A)                   # b_t on the grid, per column
        for k in 1:L
            betas[t, k] = B[k, k]
        end
        S .= B .* mask
    end
    # forward: alpha_t = P(X_1..X_{t-1} < u_k, X_t = u_k)
    S .= p0 .* mask
    for k in 2:L
        P[1] += p0[k] * betas[1, k]
    end
    for t in 1:(n - 1)
        _transition!(M, c, x, dx, t)
        A .= Wt .* S
        mul!(B, M, A)
        S .= B .* mask
        acc = 0.0
        for k in 2:L
            acc += S[k, k] * betas[t + 1, k]
        end
        P[t + 1] += acc
    end
    total = sum(P)
    (isfinite(total) && abs(total - 1) <= 0.05) ||
        error("argmax marginals defective (mass $total): raise points=")
    return P ./ total
end

end # module
