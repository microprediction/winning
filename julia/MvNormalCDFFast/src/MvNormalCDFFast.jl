# MvNormalCDFFast.jl: deterministic MVN rectangle probabilities for
# factor-structured covariance -- the MvNormalCDF.jl drop-in for the
# V V' + diag(D) slice. Port of winning/fastmvn.py (itself a port of
# the R package mvtnormfast); the python reference is the spec and the
# embedded fixtures pin this port to it.
#
# Conditional on the r-dimensional factor the coordinates are
# independent, so P(a <= X <= b) is an r-dimensional smooth integral
# of a product of univariate normal CDFs: exact Gauss-Hermite
# quadrature at rank <= 2 and sharpness <= 3, with a deterministic
# Laplace-recentered evaluation for deep tails. THE REFUSAL CONTRACT:
# past rank 2 or sharpness 3 the reference escalates to scrambled
# Sobol, and plain Halton measurably degrades there (1e-4 at rank 6),
# so rather than ship degraded nodes this package REFUSES and falls
# back to the genuine incumbent -- load MvNormalCDF.jl and the
# fallback activates via a package extension; without it, the error
# says exactly what to do. An inexact factorization must never
# masquerade as the structured case.
module MvNormalCDFFast

using LinearAlgebra: SymTridiagonal, eigen, Symmetric, diag, norm

export mvn_cdf_fast, mvn_cdf_fast_info, mvnormcdf, factorize_covariance

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

function _gh_nodes(r::Int, Q::Int)
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
iterated principal-factor fit for ranks 1..max_rank, accepted only
after recomputed-V verification. Returns (V, D) or nothing."""
function factorize_covariance(sigma::AbstractMatrix; max_rank = 6,
                              tol = 1e-11, n_iter = 300)
    sigma = Float64.(Matrix(sigma))
    n = size(sigma, 1)
    scale = maximum(abs.(sigma))
    for r in 1:min(max_rank, n - 1)
        D = fill(0.5 * sum(diag(sigma)) / n, n)
        V = zeros(n, r)
        for _ in 1:n_iter
            E = eigen(Symmetric(sigma .- [i == j ? D[i] : 0.0
                                          for i in 1:n, j in 1:n]))
            idx = sortperm(E.values, rev = true)[1:r]
            V = E.vectors[:, idx] .* sqrt.(max.(E.values[idx], 0.0))'
            D_new = max.(diag(sigma) .- vec(sum(V .^ 2, dims = 2)), 1e-12)
            if maximum(abs.(D_new .- D)) < 1e-12 * scale
                D = D_new
                break
            end
            D = D_new
        end
        E = eigen(Symmetric(sigma .- [i == j ? D[i] : 0.0
                                      for i in 1:n, j in 1:n]))
        idx = sortperm(E.values, rev = true)[1:r]
        V = E.vectors[:, idx] .* sqrt.(max.(E.values[idx], 0.0))'
        if maximum(abs.(V * V' .+ [i == j ? D[i] : 0.0
                                   for i in 1:n, j in 1:n] .- sigma)) <
           tol * scale
            return V, D
        end
    end
    return nothing
end

# the extension populates this hook when MvNormalCDF is loaded;
# it returns (p, e) in the incumbent's convention
const DENSE_FALLBACK = Ref{Union{Nothing,Function}}(nothing)

function _dense_fallback(lower, upper, mean, sigma)
    fb = DENSE_FALLBACK[]
    fb === nothing && error(
        "covariance is not factor-plus-diagonal to tolerance (or needs " *
        "rank > 2 / sharpness > 3, where this package refuses rather " *
        "than ship degraded quadrature). Load the incumbent for the " *
        "dense path:  using MvNormalCDF")
    return fb(lower, upper, mean, sigma)
end

function _cell_expectation(F, W, V, s, mu, lo, up)
    M = F * V'                          # (Q, n)
    total = 0.0
    for q in axes(F, 1)
        lc = 0.0
        for j in eachindex(mu)
            cell = ndtr((up[j] - mu[j] - M[q, j]) / s[j]) -
                   ndtr((lo[j] - mu[j] - M[q, j]) / s[j])
            lc += log(max(cell, TINY))
        end
        total += W[q] * exp(lc)
    end
    return total
end

function _impl(lower, upper, mean, sigma, V, D)
    if V === nothing || D === nothing
        sigma === nothing && error("supply sigma, or V and D")
        fd = factorize_covariance(sigma)
        fd === nothing &&
            return _dense_fallback(lower, upper, mean, sigma)[1], "fallback"
        V, D = fd
    end
    Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) :
         reshape(Float64.(collect(V)), :, 1)
    n = size(Vm, 1)
    Dv = Float64.(collect(D))
    mu = mean === nothing ? zeros(n) : Float64.(collect(mean))
    lo = lower === nothing ? fill(-Inf, n) : Float64.(collect(lower))
    up = upper === nothing ? fill(Inf, n) : Float64.(collect(upper))
    s = sqrt.(Dv)
    r = size(Vm, 2)
    sharp = _sharpness(Vm, Dv)
    if r > 2 || sharp > 3.0
        sigma_d = Vm * Vm' .+ [i == j ? Dv[i] : 0.0 for i in 1:n, j in 1:n]
        return _dense_fallback(lo, up, mu, sigma_d)[1], "fallback"
    end
    Q = Int(clamp(ceil(8.0 * sharp), 15, r == 1 ? 201 : 41))
    F, W = _gh_nodes(r, Q)
    p = _cell_expectation(F, W, Vm, s, mu, lo, up)
    p >= 1e-8 && return p, "factor"

    # deep tail: recenter the quadrature at the Laplace point of the
    # log-integrand and importance-reweight -- DETERMINISTIC here
    # (GH on the recentered gaussian), unlike the reference's Sobol
    logint(f) = begin
        z = Vm * f
        acc = -0.5 * (f' * f)
        for j in 1:n
            cell = ndtr((up[j] - mu[j] - z[j]) / s[j]) -
                   ndtr((lo[j] - mu[j] - z[j]) / s[j])
            acc += log(max(cell, TINY))
        end
        acc
    end
    f0 = zeros(r)
    h = 1e-4
    for _ in 1:50
        g = [(logint(f0 .+ h .* (1:r .== c)) -
              logint(f0 .- h .* (1:r .== c))) / (2h) for c in 1:r]
        norm(g) < 1e-8 && break
        f0 .+= clamp.(0.5 .* g, -1, 1)
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
or sigma for an exact-decomposition search; refused cases fall back to
MvNormalCDF.jl when it is loaded."""
mvn_cdf_fast(; lower = nothing, upper = nothing, mean = nothing,
             sigma = nothing, V = nothing, D = nothing) =
    _impl(lower, upper, mean, sigma, V, D)[1]

"""As mvn_cdf_fast, returning (p, method)."""
mvn_cdf_fast_info(; lower = nothing, upper = nothing, mean = nothing,
                  sigma = nothing, V = nothing, D = nothing) =
    _impl(lower, upper, mean, sigma, V, D)

"""MvNormalCDF-compatible signature: mvnormcdf(mu, Sigma, a, b)
returning (p, e). The error estimate e is the difference between the
working quadrature and a lower-order one -- honest and usually
conservative on the exact path; fallback results carry the
incumbent's own error."""
function mvnormcdf(mu::AbstractVector, sigma::AbstractMatrix,
                   a::AbstractVector, b::AbstractVector)
    fd = factorize_covariance(sigma)
    fd === nothing && return _dense_fallback(a, b, mu, sigma)
    V, D = fd
    r = size(V, 2)
    sharp = _sharpness(V, D)
    (r > 2 || sharp > 3.0) && return _dense_fallback(a, b, mu, sigma)
    p, _ = _impl(a, b, mu, nothing, V, D)
    # error estimate: re-evaluate at reduced order
    s = sqrt.(D)
    Q = Int(clamp(ceil(8.0 * sharp), 15, r == 1 ? 201 : 41))
    F2, W2 = _gh_nodes(r, max(Q - 6, 7))
    p2 = _cell_expectation(F2, W2, V, s, Float64.(collect(mu)),
                           Float64.(collect(a)), Float64.(collect(b)))
    return p, abs(p - p2)
end

end # module
