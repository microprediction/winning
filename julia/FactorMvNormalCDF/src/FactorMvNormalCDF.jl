# FactorMvNormalCDF.jl: deterministic MVN rectangle probabilities for
# factor-structured covariance, the companion of MvNormalCDF.jl for the
# V V' + diag(D) slice. Port of winning/fastmvn.py (itself a port of
# the R package mvtnormfast); the python reference is the spec and the
# embedded fixtures pin this port to it.
#
# Conditional on the r-dimensional factor the coordinates are
# independent, so P(a <= X <= b) is an r-dimensional smooth integral
# of a product of univariate normal CDFs: exact Gauss-Hermite
# quadrature at rank <= 2 and sharpness <= 3, with a deterministic
# Laplace-recentered evaluation for deep tails. The exact path claims
# only what it computes exactly. Past rank 2 or sharpness 3 the
# reference escalates to scrambled Sobol and plain Halton measurably
# degrades there (1e-4 at rank 6), so those cases, and any covariance
# that does not verify as factor-plus-diagonal, are delegated to
# MvNormalCDF.mvnormcdf, a dependency, and every call is answered. An
# inexact factorization never masquerades as the structured case.
module FactorMvNormalCDF

using LinearAlgebra: SymTridiagonal, eigen, Symmetric, diag, norm
import MvNormalCDF

export mvn_cdf_fast, mvn_cdf_fast_info, mvnormcdf_factor, factorize_covariance

const TINY = 1e-300

# ---- normal cdf (the winning core's series + erfcx recipe) ---------

function _erf_series(x::Float64)
    s = x
    t = x
    for n in 1:119
        t *= -x * x / n
        s += t / (2n + 1)
        abs(t) < 1e-19 * abs(s) && break
    end
    return (2 / sqrt(pi)) * s
end

function _erfcx(x::Float64)
    cf = 0.0
    for k in 60:-1:1
        cf = (k / 2) / (x + cf)
    end
    return 1 / (sqrt(pi) * (x + cf))
end

# ---- normal cdf: Cephes ndtr (Moshier, public domain; the same
# rational approximations scipy's ndtr compiles). The series+erfcx
# recipe this replaces is kept as _ndtr_series, the test oracle. ----

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

@inline function _polevl(x::Float64, C)
    y = C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _p1evl(x::Float64, C)
    # leading coefficient 1 implied: y = ((x + C[1]) x + C[2]) x ...
    y = x + C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _cephes_erf(x::Float64)
    abs(x) > 1.0 && return 1.0 - _cephes_erfc(x)
    z = x * x
    return x * _polevl(z, _CT) / _p1evl(z, _CU)
end

@inline function _cephes_erfc(a::Float64)
    x = abs(a)
    x < 1.0 && return 1.0 - _cephes_erf(a)
    z = -a * a
    z < -_MAXLOG && return a < 0 ? 2.0 : 0.0
    z = exp(z)
    if x < 8.0
        p = _polevl(x, _CP)
        q = _p1evl(x, _CQ)
    else
        p = _polevl(x, _CR)
        q = _p1evl(x, _CS)
    end
    y = (z * p) / q
    a < 0 && (y = 2.0 - y)
    return y
end

function ndtr(a::Float64)
    isinf(a) && return a > 0 ? 1.0 : 0.0
    x = a * _SQRTH
    z = abs(x)
    if z < 1.0
        return 0.5 + 0.5 * _cephes_erf(x)
    end
    y = 0.5 * _cephes_erfc(z)
    return x > 0 ? 1.0 - y : y
end

function _ndtr_series(z::Float64)
    isinf(z) && return z > 0 ? 1.0 : 0.0
    x = z / sqrt(2)
    x >= 2.5 && return 1 - 0.5 * _erfcx(x) * exp(-x * x)
    x <= -2.5 && return 0.5 * _erfcx(-x) * exp(-x * x)
    return 0.5 * (1 + _erf_series(x))
end

function _gh1(Q::Int)
    T = SymTridiagonal(zeros(Q), sqrt.(1.0:(Q - 1)))
    E = eigen(T)
    w = E.vectors[1, :] .^ 2
    return E.values, w ./ sum(w)
end

# Rank two keeps the 15-point floor so ordinary inputs cost what they
# did; rank one floors at 61 (n x Q work, negligible once cached).
_gh_order(r, sharp) = Int(clamp(ceil(15.0 * sharp^2), r == 1 ? 61 : 15,
                                r == 1 ? 201 : 41))

const _GH_CACHE = Dict{Tuple{Int,Int},Tuple{Matrix{Float64},Vector{Float64}}}()
const _GH_LOCK = ReentrantLock()

# cached: the rule depends on (r, Q) alone, and its eigensolve cost more
# than the integral it serves
function _gh_nodes(r::Int, Q::Int)
    lock(_GH_LOCK) do
        get!(() -> _gh_nodes_build(r, Q), _GH_CACHE, (r, Q))
    end
end

function _gh_nodes_build(r::Int, Q::Int)
    # r = 0 is the EMPTY PRODUCT: one node of weight 1 with no columns,
    # so an (n, 0) loading matrix reduces exactly to the independent
    # product. Every r != 1 was built as RANK TWO below, so r = 0 made a
    # spurious Q^2 x 2 node matrix and then failed multiplying it by a
    # 0 x n loading transpose (#68). Callers route r > 2 to the dense
    # fallback, so 0, 1 and 2 are the whole domain here.
    if r == 0
        return Matrix{Float64}(undef, 1, 0), [1.0]
    end
    r < 0 && throw(ArgumentError("_gh_nodes needs r >= 0; got $r"))
    x, w = _gh1(Q)
    if r == 1
        return reshape(x, :, 1), w
    end
    F = zeros(Q * Q, 2)
    W = ones(Q * Q)
    t = 0
    for ia in 1:Q, ib in 1:Q          # first coordinate slowest
        t += 1
        F[t, 1] = x[ia]
        F[t, 2] = x[ib]
        W[t] = w[ia] * w[ib]
    end
    keep = W .> 1e-12 * maximum(W)
    return F[keep, :], W[keep] ./ sum(W[keep])
end

_sharpness(V, D) = maximum(sqrt.(vec(sum(V .^ 2, dims = 2))) ./
                           sqrt.(max.(D, TINY)))

"""Exact V V' + diag(D) decomposition of sigma, if one exists:
iterated principal-factor fit for ranks 0..max_rank on the CORRELATION
matrix, scaled back, accepted only after recomputed-V verification.
Returns (V, D) or nothing.

The search used an absolute residual floor (1e-12) and a tolerance
relative to the largest entry, so `1e-12 * S` was rejected while `S` was
accepted (#368), and a 0.9 correlation beside an unrelated 1e16 variance
counted as negligible (#132). Rank zero is tried first: a diagonal sigma
has no factors, and the rank-one search represented diag(1,1,1,1,1e8)
with an artificial loading of 9487, which priced its independent
rectangle 5.6e-4 low (#132)."""
function factorize_covariance(sigma::AbstractMatrix; max_rank = 6,
                              tol = 1e-11, n_iter = 300)
    sigma = Float64.(Matrix(sigma))
    n = size(sigma, 1)
    (size(sigma, 2) == n && all(isfinite, sigma)) || return nothing
    d = diag(sigma)
    any(<(0.0), d) && return nothing
    pos = d .> 0.0
    for i in 1:n, j in 1:n
        if (!pos[i] || !pos[j]) && sigma[i, j] != 0.0
            return nothing
        end
    end
    m = count(pos)
    sd = sqrt.(d[pos])
    R = sigma[pos, pos] ./ (sd * sd')
    offmax = m == 0 ? 0.0 :
        maximum(abs(R[i, j]) for i in 1:m, j in 1:m if i != j; init = 0.0)
    if m == 0 || offmax <= tol
        return zeros(n, 0), copy(d)
    end
    dR = diag(R)
    dg(D) = [i == j ? D[i] : 0.0 for i in 1:m, j in 1:m]
    for r in 1:min(max_rank, m - 1)
        D = fill(0.5, m)
        V = zeros(m, r)
        for _ in 1:n_iter
            E = eigen(Symmetric(R .- dg(D)))
            idx = sortperm(E.values, rev = true)[1:r]
            V = E.vectors[:, idx] .* sqrt.(max.(E.values[idx], 0.0))'
            D_new = max.(dR .- vec(sum(V .^ 2, dims = 2)), 1e-12)
            if maximum(abs.(D_new .- D)) < 1e-12
                D = D_new
                break
            end
            D = D_new
        end
        E = eigen(Symmetric(R .- dg(D)))
        idx = sortperm(E.values, rev = true)[1:r]
        V = E.vectors[:, idx] .* sqrt.(max.(E.values[idx], 0.0))'
        if maximum(abs.(V * V' .+ dg(D) .- R)) < tol
            Vf = zeros(n, r)
            Vf[pos, :] = V .* sd
            Df = zeros(n)
            Df[pos] = D .* d[pos]
            return Vf, Df
        end
    end
    return nothing
end

"""
    _as_covariance(sigma) -> Matrix{Float64}

A finite, square, symmetric, positive-semidefinite matrix, or an
ArgumentError; returns the symmetrised matrix. A plain matrix reaching
MvNormalCDF is read through its LOWER TRIANGLE only, so an asymmetric
`sigma` and its transpose priced two different Gaussians -- 0.1732 and
0.2270 on the same orthant (#359). Same tolerances as the python
reference.
"""
function _as_covariance(sigma)
    C = Float64.(Matrix(sigma))
    size(C, 1) == size(C, 2) || throw(ArgumentError(
        "sigma must be a square matrix; got size $(size(C))"))
    all(isfinite, C) || throw(ArgumentError("sigma contains NaN or inf"))
    isempty(C) && return C
    asym = maximum(abs.(C .- C'))
    if asym > 1e-8 * max(maximum(abs.(C)), 1e-300)
        throw(ArgumentError(
            "sigma is not symmetric (max asymmetry $asym); pass " *
            "(sigma + sigma')/2 if the asymmetry is numerical noise"))
    end
    C = 0.5 .* C .+ 0.5 .* C'
    n = size(C, 1)
    lam = minimum(eigen(Symmetric(C)).values)
    if lam < -1e-8 * max(sum(diag(C) ./ n), 1e-300)
        throw(ArgumentError(
            "sigma is not positive semidefinite (min eigenvalue $lam); " *
            "it is not a covariance matrix"))
    end
    return C
end

# Delegation to the incumbent for every case outside the exact path;
# returns (p, e) in MvNormalCDF's convention. Keyword arguments (m, rng)
# are forwarded so a caller controls the QMC budget there.
"""
    _bound_vector(x, default, n) -> Vector{Float64}

`winning.fastmvn` is the specification, and it broadcasts a scalar bound
to every coordinate (`np.broadcast_to`); the R port does the same with
`rep_len`. This port called `collect` on the argument, and a scalar
`Float64` is not iterable, so the ordinary joint-CDF spelling
`mvn_cdf_fast(V=V, D=D, upper=0.0)` threw before evaluating anything, on
both the structured and the delegated dense path (#184).
"""
function _bound_vector(x, default::Float64, n::Int, name::AbstractString)
    x === nothing && return fill(default, n)
    x isa Number && return fill(Float64(x), n)
    v = Float64.(collect(x))
    length(v) == 1 && return fill(v[1], n)
    length(v) == n || throw(ArgumentError(
        "$name has length $(length(v)) but the problem has $n " *
        "coordinates; pass a scalar or one bound per coordinate"))
    return v
end

"""
    _log_ndtr(x) -> Float64

log Phi(x) without underflow: log of the Cephes ndtr down to -30, the
erfcx form `log(erfcx(-x/sqrt 2)/2) - x^2/2` in the far
lower tail, and log1p(-Phi(-x)) above zero.
"""
function _log_ndtr(x::Float64)
    x == Inf && return 0.0
    x == -Inf && return -Inf
    x > 0.0 && return log1p(-ndtr(-x))
    # Cephes ndtr is relatively accurate down the whole left tail until it
    # underflows near -37.5; the 60-term erfcx continued fraction is only
    # needed past that, and cost 6x the integral when used from -5
    x > -30.0 && return log(ndtr(x))
    return log(0.5 * _erfcx(-x * _SQRTH)) - 0.5 * x * x
end

"""
    _log_cell(hi, lo, sd) -> Float64

One coordinate's LOG conditional cell mass, `sd == 0` included.

In the log domain, on the side of zero where nothing cancels. The mass
used to be `ndtr(hi/sd) - ndtr(lo/sd)`: in the upper tail both round to
1, so `P(9 < Z <= 10) = 1.13e-19` became 0 and then the `TINY` floor
(#196). Upper-tail intervals are reflected to the lower tail, so an
exactly empty cell is `-Inf` and nothing needs a floor.

A coordinate with zero idiosyncratic variance is DETERMINISTIC given the
factor draw, so its cell is an INDICATOR (#206). `hi` and `lo` are
already shifted by the mean and the factor term.
"""
function _log_cell(hi::Float64, lo::Float64, sd::Float64)
    sd > 0.0 || return (lo <= 0.0 && 0.0 <= hi) ? 0.0 : -Inf
    lo == -Inf && return _log_ndtr(hi / sd)    # the common one-sided cell
    b = hi / sd
    a = lo / sd
    if a > 0.0
        a, b = -b, -a
    end
    a >= b && return -Inf
    if b <= 0.0
        lb = _log_ndtr(b)
        return lb + log1p(-exp(_log_ndtr(a) - lb))
    end
    return log1p(-(ndtr(a) + ndtr(-b)))
end

"""
    _check_bounds(lo, up)

Refuse a reversed rectangle `{lo <= x <= up}`: the event is EMPTY, and
every port used to clamp its negative conditional cell to the underflow
floor and return a finite probability (#235). `mvtnorm::pmvnorm`, which
the R port is a drop-in for, raises on reversed bounds.

What `lower == upper` means is decided by `_drop_constants`, because it
depends on the covariance (#414).
"""
function _check_bounds(lo, up)
    for i in eachindex(lo, up)
        (isnan(lo[i]) || isnan(up[i])) && throw(ArgumentError(
            "lower and upper must not contain NaN"))
        if lo[i] > up[i]
            throw(ArgumentError(
                "lower must not exceed upper: coordinate $i has " *
                "lower=$(lo[i]) > upper=$(up[i]), so the rectangle is " *
                "empty. Check the argument order."))
        end
    end
    return nothing
end

"""
    _drop_constants(var, mu, lo, up) -> (keep, answer)

A coordinate of zero marginal variance is the constant `mu_i`: it
contributes the INDICATOR `lo_i <= mu_i <= up_i`, inclusive, and is
independent of the rest. A miss is an empty rectangle and a hit leaves
the integral. `lower == upper` is then zero mass only where the
variance is positive; it used to be zero mass everywhere, so an atom at
the bound beside N(0, 1) priced 0 where the answer is 0.5 (#414).
`answer` is `nothing` or the `(p, method)` result.
"""
function _drop_constants(var, mu, lo, up)
    const_ = var .== 0.0
    for i in eachindex(var)
        if const_[i] && (mu[i] < lo[i] || mu[i] > up[i])
            return const_, (0.0, "outside-support")
        end
    end
    keep = .!const_
    for i in eachindex(var)
        keep[i] && lo[i] == up[i] && return keep, (0.0, "degenerate-rectangle")
    end
    any(keep) || return keep, (1.0, "factor")
    return keep, nothing
end

# One coordinate: the marginal is N(mu, var) whatever the factor
# decomposition, so no quadrature (#395).
_univariate(lo, up, mu, var) =
    (exp(_log_cell(up - mu, lo - mu, sqrt(var))), "factor")

function _dense_fallback(lower, upper, mean, sigma; kwargs...)
    n = size(sigma, 1)
    mu = mean === nothing ? zeros(n) : Float64.(collect(mean))
    lo = _bound_vector(lower, -Inf, n, "lower")
    up = _bound_vector(upper, Inf, n, "upper")
    _check_bounds(lo, up)
    S = Float64.(Matrix(sigma))
    keep, done = _drop_constants(diag(S), mu, lo, up)
    done === nothing || return (done[1], 0.0)
    if count(keep) == 1
        k = findfirst(keep)
        return (_univariate(lo[k], up[k], mu[k], S[k, k])[1], 0.0)
    end
    return MvNormalCDF.mvnormcdf(mu[keep], S[keep, keep], lo[keep],
                                 up[keep]; kwargs...)
end

function _cell_expectation(F, W, V, s, mu, lo, up)
    M = F * V'                          # (Q, n)
    total = 0.0
    for q in axes(F, 1)
        lc = 0.0
        for j in eachindex(mu)
            lc += _log_cell(up[j] - mu[j] - M[q, j],
                            lo[j] - mu[j] - M[q, j], s[j])
        end
        total += W[q] * exp(lc)
    end
    return total
end

function _impl(lower, upper, mean, sigma, V, D; kwargs...)
    if V === nothing || D === nothing
        sigma === nothing && error("supply sigma, or V and D")
        sigma = _as_covariance(sigma)
        nn = size(sigma, 1)
        lo0 = _bound_vector(lower, -Inf, nn, "lower")
        up0 = _bound_vector(upper, Inf, nn, "upper")
        _check_bounds(lo0, up0)
        fd = factorize_covariance(sigma)
        if fd === nothing
            mu0 = mean === nothing ? zeros(nn) : Float64.(collect(mean))
            keep, done = _drop_constants(diag(sigma), mu0, lo0, up0)
            done === nothing || return done
            return _dense_fallback(lo0, up0, mu0, sigma; kwargs...)[1], "fallback"
        end
        V, D = fd
    end
    Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) :
         reshape(Float64.(collect(V)), :, 1)
    n = size(Vm, 1)
    Dv = D isa Number ? fill(Float64(D), n) : Float64.(collect(D))
    length(Dv) == 1 && (Dv = fill(Dv[1], n))
    length(Dv) == n || throw(ArgumentError(
        "D has length $(length(Dv)) but the problem has $n coordinates"))
    mu = mean === nothing ? zeros(n) : _bound_vector(mean, 0.0, n, "mean")
    lo = _bound_vector(lower, -Inf, n, "lower")
    up = _bound_vector(upper, Inf, n, "upper")
    _check_bounds(lo, up)
    var = vec(sum(Vm .^ 2, dims = 2)) .+ Dv
    keep, done = _drop_constants(var, mu, lo, up)
    done === nothing || return done
    if !all(keep)
        Vm = Vm[keep, :]; Dv = Dv[keep]; mu = mu[keep]
        lo = lo[keep]; up = up[keep]; var = var[keep]
    end
    n = length(Dv)
    n == 1 && return _univariate(lo[1], up[1], mu[1], var[1])
    # a factor no coordinate loads on integrates out exactly
    Vm = Vm[:, [any(!=(0.0), Vm[:, c]) for c in 1:size(Vm, 2)]]
    s = sqrt.(Dv)
    r = size(Vm, 2)
    if r == 0
        return exp(sum(_log_cell(up[j] - mu[j], lo[j] - mu[j], s[j])
                       for j in 1:n)), "factor"
    end
    sharp = _sharpness(Vm, Dv)
    if r > 2 || sharp > 3.0
        sigma_d = Vm * Vm' .+ [i == j ? Dv[i] : 0.0 for i in 1:n, j in 1:n]
        return _dense_fallback(lo, up, mu, sigma_d; kwargs...)[1], "fallback"
    end
    # The order grows like sharpness SQUARED, not 8 x sharpness: the
    # linear rule left 0.2% relative error on a correlation-0.8 orthant
    # (#132). Same rule as the python reference.
    Q = _gh_order(r, sharp)
    F, W = _gh_nodes(r, Q)
    p = _cell_expectation(F, W, Vm, s, mu, lo, up)
    p >= 1e-8 && return p, "factor"

    # deep tail: recenter the quadrature at the Laplace point of the
    # log-integrand and importance-reweight -- DETERMINISTIC here
    # (GH on the recentered gaussian), unlike the reference's Sobol
    # `let` rebinds the captures: Vm, mu, lo, up and n are reassigned
    # above (constants dropped), and a closure over a reassigned variable
    # boxes it, which made this loop type-unstable and 6x slower
    logint = let Vm = Vm, mu = mu, lo = lo, up = up, s = s, n = n
        f -> begin
            z = Vm * f
            acc = -0.5 * (f' * f)
            for j in 1:n
                acc += _log_cell(up[j] - mu[j] - z[j], lo[j] - mu[j] - z[j],
                                 s[j])
            end
            acc
        end
    end
    # backtracking ascent on the smooth log-integrand (python _ascend)
    f0 = zeros(r)
    h = 1e-4
    val = logint(f0)
    for _ in 1:100
        g = [(logint(f0 .+ h .* (1:r .== c)) -
              logint(f0 .- h .* (1:r .== c))) / (2h) for c in 1:r]
        (all(isfinite, g) && norm(g) >= 1e-8) || break
        step = 1.0
        moved = false
        while step > 1e-12
            cand = f0 .+ clamp.(step .* g, -1, 1)
            cv = logint(cand)
            if cv > val
                f0, val, moved = cand, cv, true
                break
            end
            step /= 2
        end
        moved || break
    end
    tau = 1.5
    Fh, Wh = _gh_nodes(r, r == 1 ? 201 : 41)
    total = 0.0
    for q in axes(Fh, 1)
        f = f0 .+ tau .* vec(Fh[q, :])
        # E_{x~N(0,I)}[ exp(logint(f0 + tau x) + |x|^2/2) tau^r ]
        lw = logint(f) + 0.5 * sum(abs2, vec(Fh[q, :])) + r * log(tau)
        total += Wh[q] * exp(lw)
    end
    return total, "factor-recentered"
end

"""P(lower <= X <= upper), X ~ N(mean, V V' + diag(D)). Supply (V, D),
or sigma for an exact-decomposition search. Cases outside the exact
path go to MvNormalCDF.mvnormcdf; keyword arguments (m, rng) are
forwarded to it."""
mvn_cdf_fast(; lower = nothing, upper = nothing, mean = nothing,
             sigma = nothing, V = nothing, D = nothing, kwargs...) =
    _impl(lower, upper, mean, sigma, V, D; kwargs...)[1]

"""As mvn_cdf_fast, returning (p, method) with method one of "factor",
"factor-recentered" or "fallback"."""
mvn_cdf_fast_info(; lower = nothing, upper = nothing, mean = nothing,
                  sigma = nothing, V = nothing, D = nothing, kwargs...) =
    _impl(lower, upper, mean, sigma, V, D; kwargs...)

"""MvNormalCDF's signature under this package's own name:
mvnormcdf_factor(mu, Sigma, a, b; kwargs...) returning (p, e). On the
exact path e is the difference between the working quadrature and a
lower-order one, usually conservative; delegated results carry
MvNormalCDF's own estimate, and kwargs (m, rng) go to it."""
function mvnormcdf_factor(mu::AbstractVector, sigma::AbstractMatrix,
                          a::AbstractVector, b::AbstractVector; kwargs...)
    sigma = _as_covariance(sigma)
    fd = factorize_covariance(sigma)
    fd === nothing && return _dense_fallback(a, b, mu, sigma; kwargs...)
    V, D = fd
    r = size(V, 2)
    sharp = r == 0 ? 0.0 : _sharpness(V, D)
    (r > 2 || sharp > 3.0) &&
        return _dense_fallback(a, b, mu, sigma; kwargs...)
    p, method = _impl(a, b, mu, nothing, V, D)
    r > 0 || return p, 0.0
    # error estimate: re-evaluate at reduced order
    s = sqrt.(D)
    Q = _gh_order(r, sharp)
    F2, W2 = _gh_nodes(r, max(Q - 6, 7))
    p2 = _cell_expectation(F2, W2, V, s, Float64.(collect(mu)),
                           Float64.(collect(a)), Float64.(collect(b)))
    return p, abs(p - p2)
end

end # module
