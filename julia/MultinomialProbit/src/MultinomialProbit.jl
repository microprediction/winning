# MultinomialProbit.jl: the model no Julia package has. Factor
# multinomial probit -- utilities U_tj = mu_tj + (V f_t)_j + z_tj,
# f_t ~ N(0, I_r), z ~ N(0, I), choice = argmax -- with TWO engines
# behind one interface:
#
#   method = :exact   the factor-conditional product-of-CDFs likelihood
#                     with ANALYTIC score (port of winning/likelihood.py,
#                     itself translated from r/mlogitfast); smooth,
#                     deterministic, no simulation.
#   method = :ghk     the field's incumbent, for reference and for
#                     dense problems: the GHK sequential importance
#                     sampler (port of the rust core's ghk_prob_one),
#                     common-random-numbers so the simulated likelihood
#                     is a deterministic function of the parameters.
#
# Covariance parameterization as in the python reference: rank-r
# loadings with a zero reference row and a strictly-lower-triangular
# free block, unit idiosyncratic variances -- at J = 4 alternatives and
# r = 2 this spans the identified differenced covariance with the same
# dof count as the differenced-Cholesky parameterization of R's mlogit.
# Ranks above J - 2 are not identified and are refused (#201).
#
# Sharpness rule (third+ appearance in the winning repo): past
# sqrt(2) max_j ||v_j|| / sqrt(min D) = 3 the factor integrand is a
# near-step, Gauss-Hermite under-integrates at any order and an
# optimizer will EXPLOIT THE HOLES; the evaluation escalates to Halton
# nodes, the dependency-free escalation of the R port.
module MultinomialProbit

import LinearAlgebra
using LinearAlgebra: SymTridiagonal, eigen, cholesky, Symmetric
using Random: Xoshiro

export MNProbit, max_identified_rank, fit!, loglikelihood, predict_proba, coef, vcov,
    stderror, score_matrix, loglik_hessian,
    choice_loglik_and_score, ghk_choice_prob

const TINY = 1e-300

# ---- normal cdf / inverse (the winning JS/Julia core's recipes) ----

function _erf_series(x::Real)          # Real, not Float64: logndtr
    # below is called on ForwardDiff duals, and this is pure arithmetic
    s = x
    t = x
    for n in 1:119
        t *= -x * x / n
        s += t / (2n + 1)
        abs(t) < 1e-19 * abs(s) && break
    end
    return (2 / sqrt(pi)) * s
end

function _erfcx(x::Real)
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

@inline function _polevl(x::Real, C)
    y = zero(x) + C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _p1evl(x::Real, C)
    # leading coefficient 1 implied: y = ((x + C[1]) x + C[2]) x ...
    y = x + C[1]
    @inbounds for i in 2:length(C)
        y = y * x + C[i]
    end
    return y
end

@inline function _cephes_erf(x::Real)
    abs(x) > 1.0 && return 1.0 - _cephes_erfc(x)
    z = x * x
    return x * _polevl(z, _CT) / _p1evl(z, _CU)
end

@inline function _cephes_erfc(a::Real)
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

function ndtr(a::Real)
    isinf(a) && return a > 0 ? 1.0 : 0.0
    x = a * _SQRTH
    z = abs(x)
    if z < 1.0
        return 0.5 + 0.5 * _cephes_erf(x)
    end
    y = 0.5 * _cephes_erfc(z)
    return x > 0 ? 1.0 - y : y
end

function _ndtr_series(z::Real)
    x = z / sqrt(2)
    x >= 2.5 && return 1 - 0.5 * _erfcx(x) * exp(-x * x)
    x <= -2.5 && return 0.5 * _erfcx(-x) * exp(-x * x)
    return 0.5 * (1 + _erf_series(x))
end

function ndtri(p::Float64)
    a = (-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742)
    pl = 0.02425
    if p < pl
        q = sqrt(-2 * log(p))
        x = (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p > 1 - pl
        return -ndtri(1 - p)
    else
        q = p - 0.5
        r2 = q * q
        x = (((((a[1] * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * r2 + a[6]) * q /
            (((((b[1] * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + b[5]) * r2 + 1)
    end
    # one Halley polish step (Acklam alone is ~1e-9; the GHK recursion
    # feeds ndtri output back through ndtr, so polish to near machine)
    e = ndtr(x) - p
    u = e * sqrt(2 * pi) * exp(0.5 * x * x)
    return x - u / (1 + 0.5 * x * u)
end

function _gh1(Q::Int)
    T = SymTridiagonal(zeros(Q), sqrt.(1.0:(Q - 1)))
    E = eigen(T)
    w = E.vectors[1, :] .^ 2
    return E.values, w ./ sum(w)
end

"""The first `n` primes. The Halton table used to be a fixed tuple of
eight, so a sharp rank-8 likelihood (r + 1 = 9 dimensions) raised a
BoundsError at `primes[9]` (#396)."""
function _first_primes(n::Int)
    ps = Int[]
    c = 2
    while length(ps) < n
        all(p -> c % p != 0, ps) && push!(ps, c)
        c += 1
    end
    return ps
end

function _halton_normal(r::Int, n::Int)
    primes = _first_primes(r)
    F = zeros(n, r)
    for c in 1:r
        b = primes[c]
        for row in 1:n
            i = row + 20
            f = 1.0 / b
            h = 0.0
            while i > 0
                h += f * (i % b)
                i = div(i, b)
                f /= b
            end
            F[row, c] = ndtri(clamp(h, 1e-12, 1 - 1e-12))
        end
    end
    return F, fill(1.0 / n, n)
end

"""log of the standard normal CDF, tail-stable. `log(max(ndtr(a), TINY))`
underflows to the same -690.78 for every a below about -37, which made the
likelihood objective FLAT in the tail and its score exactly zero (#270)."""
# _erfcx here is the continued fraction, which has NOT converged for
# small x: at x = 1/sqrt(2), where a z <= -1 split would first call it,
# it is wrong by 5e-7 relative and that error lands straight in the
# log-likelihood. _ndtr_series already knows the right crossover, 2.5,
# so use the same one rather than inventing a second.
const _LOGNDTR_CUT = 2.5 * sqrt(2)

function logndtr(z::Real)
    # cephes ndtr in the ordinary range, not _ndtr_series: the series
    # runs up to 119 terms and this is called on every node of every
    # Newton step of every likelihood evaluation, which made an
    # end-to-end fit so slow it looked like a hang. Above the cut the
    # CDF is O(1e-4) or larger, so log of it loses nothing.
    if z > _LOGNDTR_CUT
        return log1p(-ndtr(-z))                  # the complement is tiny
    elseif z >= -_LOGNDTR_CUT
        return log(ndtr(z))
    end
    x = -z / sqrt(2)                             # x >= 2.5: CF is good here
    return log(0.5 * _erfcx(x)) - 0.5 * z * z
end

"""log of phi(a)/Phi(a), the inverse Mills ratio, stable in the far tail."""
_log_mills(a::Real) = -0.5 * a * a - 0.5 * log(2 * pi) - logndtr(a)

"""Laplace-tilt the chosen alternative's own-noise quadrature.

The fixed Gauss-Hermite rule samples z where the PRIOR has its mass, and
for a large observed contrast the integrand's mass is far outside it: at
a gap of -20 the mode is at z = 10 and a 7-node rule reaches |z| < 3.8.
With g(z) = -z^2/2 + sum_j logPhi(A_j(z)),

    g'(z)  = -z + sum_j (s_k/s_j) lam(A_j)
    g''(z) = -1 - sum_j (s_k/s_j)^2 lam(A_j)(A_j + lam(A_j))

and lam(a)(a + lam(a)) = -lam'(a) > 0 for every a, so g'' <= -1: g is
strictly concave, its mode is unique and Newton converges from anywhere.
The change of measure is undone exactly by log sigma - z^2/2 + x^2/2, so
the shift moves the NODES and not the integrand (#270)."""
# The tilt is a choice of NODES, so it must not carry derivative
# information: if duals flowed through the Newton solve, ForwardDiff
# would return the derivative of the re-tilted objective while the
# analytic score -- correctly -- returns the quadrature of the true
# derivative, and the two would disagree by a quadrature-order amount.
# Stripping to the value makes the shift a constant, so dual-mode
# differentiation of this code reproduces the analytic score exactly.
#
# Done WITHOUT a ForwardDiff dependency, and deliberately not through
# the package extension: correctness must not depend on an extension
# having been loaded. A dual carries its value in a `value` field, and
# the recursion peels nested duals (forward-over-forward) down to the
# float underneath.
"""(F, W): nodes over (factor^r, own-noise); Halton past sharpness 3
(the dependency-free escalation of the R port)."""
function nodes_for_likelihood(r::Int; Qf = 7, Qz = 7, sharp = 0.0)
    sharp > 3.0 && return _halton_normal(r + 1, 2^10)
    xf, wf = _gh1(Qf)
    xz, wz = _gh1(Qz)
    dims = ntuple(_ -> Qf, r)
    npts = Qf^r * Qz
    F = zeros(npts, r + 1)
    W = ones(npts)
    t = 0
    for iz in 1:Qz
        for tup in Iterators.product(ntuple(_ -> 1:Qf, r)...)
            t += 1
            for c in 1:r
                F[t, c] = xf[tup[c]]
                W[t] *= wf[tup[c]]
            end
            F[t, r + 1] = xz[iz]
            W[t] *= wz[iz]
        end
    end
    keep = W .> 1e-10 * maximum(W)
    F, W = F[keep, :], W[keep]
    return F, W ./ sum(W)
end

"""Validate a choice vector against T observations and J alternatives
and return it as `Vector{Int}`. Every label must be a finite, exactly
integral number in 1..J: an integer-valued float such as `1.0` is
accepted (the Python and R contract), a fractional `1.5`, `NaN`,
`missing` or an out-of-range label is refused (#194)."""
function _check_choice(choice::AbstractVector, T::Integer, J::Integer)
    length(choice) == T || throw(ArgumentError(
        "choice must have one entry per observation: got " *
        string(length(choice)) * " for " * string(T) * " rows of mu"))
    out = Vector{Int}(undef, T)
    for (i, c) in enumerate(choice)
        ok = (c isa Integer) || (c isa Real && isfinite(c) && isinteger(c))
        ok || throw(ArgumentError(
            "choice[" * string(i) * "] = " * string(c) *
            " is not an integer alternative index"))
        (1 <= c <= J) || throw(ArgumentError(
            "choice[" * string(i) * "] = " * string(c) * " is outside 1.." *
            string(J) * "; it would be dropped in silence, which raises " *
            "the log-likelihood because there is less of it"))
        out[i] = Int(c)
    end
    return out
end

"""Idiosyncratic variances as a `Vector{Float64}` of length exactly J.
An overlong `D` used to be accepted: the integrand reads only `D[1:J]`,
but the sharpness dispatcher took `minimum` over ALL entries, so an
unused tiny trailing entry switched Gauss-Hermite to Halton and moved
the likelihood (#439)."""
function _check_D(D, J::Integer)
    D === nothing && return ones(J)
    Dv = Float64.(collect(D))
    length(Dv) == J || throw(DimensionMismatch(
        "D must have one idiosyncratic variance per alternative: got " *
        string(length(Dv)) * " for J = " * string(J)))
    return Dv
end

"""Log-likelihood of observed argmax choices with the ANALYTIC score.
mu: (T, J) utilities; V: (J, r) loadings; choice: 1-based indices.
Returns (loglik, dmu, dV). Port of winning/likelihood.py."""
function choice_loglik_and_score(mu::AbstractMatrix, V::AbstractMatrix,
                                 choice::AbstractVector;
                                 D = nothing, Qf = 7, Qz = 7,
                                 nodes = nothing, per_obs = false)
    T, J = size(mu)
    r = size(V, 2)
    # Every observation must be accounted for. The loop below walks the
    # LEGAL labels and gathers the rows matching each, so a row whose
    # choice is outside 1..J is never visited: it contributed nothing to
    # the log-likelihood and a zero row to the score, and the fit
    # silently optimised a SUBSET while reporting it as the whole. This
    # port also accepted a choice vector SHORTER than T, dropping the
    # tail without a word (#194).
    choice = _check_choice(choice, T, J)
    Dv = _check_D(D, J)
    s = sqrt.(Dv)
    V = V .- sum(V, dims = 1) ./ J          # gauge: differences decide
    ET = promote_type(eltype(mu), eltype(V), Float64)
    sharp = sqrt(2) * maximum(sqrt.(vec(sum(V .^ 2, dims = 2)))) /
            sqrt(minimum(Dv))
    F, W = nodes === nothing ?
           nodes_for_likelihood(r; Qf = Qf, Qz = Qz, sharp = sharp) : nodes
    Fq = F[:, 1:r]
    zq = F[:, r + 1]
    Q = length(W)
    Vf = Fq * V'                             # (Q, J)

    loglik = zero(ET)
    dmu = zeros(ET, T, J)
    dV = zeros(ET, J, r)
    GV = per_obs ? zeros(ET, T, J, r) : zeros(ET, 0, 0, 0)
    for k in 1:J
        idx = findall(==(k), choice)
        isempty(idx) && continue
        rivals = [j for j in 1:J if j != k]
        Ti = length(idx)
        acc = zeros(ET, Ti, Q)
        A = Dict{Int,Matrix{ET}}()
        logPhi = Dict{Int,Matrix{ET}}()
        for j in rivals
            shift = Vf[:, k] .- Vf[:, j] .+ zq .* s[k]      # (Q,)
            Aj = ((mu[idx, k] .- mu[idx, j]) .+ shift') ./ s[j]  # (Ti, Q)
            A[j] = Aj
            lp = logndtr.(Aj)
            logPhi[j] = lp
            acc .+= lp
        end
        m = vec(maximum(acc, dims = 2))
        pw = exp.(acc .- m) .* W'
        rs = vec(sum(pw, dims = 2))
        loglik += sum(m .+ log.(max.(rs, TINY)))
        omega = pw ./ rs
        for j in rivals
            lam = exp.(-0.5 .* A[j] .^ 2 .- 0.5 * log(2 * pi) .- logPhi[j])
            wl = omega .* lam ./ s[j]
            g = vec(sum(wl, dims = 2))
            dmu[idx, k] .+= g
            dmu[idx, j] .-= g
            H = wl * Fq                       # (Ti, r)
            hc = vec(sum(H, dims = 1))
            dV[k, :] .+= hc
            dV[j, :] .-= hc
            if per_obs
                GV[idx, k, :] .+= H
                GV[idx, j, :] .-= H
            end
        end
    end
    per_obs && return loglik, dmu, dV, GV
    return loglik, dmu, dV
end

"""P(alternative k chosen | mu row) by GHK sequential importance
sampling on the differenced covariance, common random numbers via the
seed. Port of the rust core's ghk_prob_one_factor; Sigma = V V' + diag(D).

`D` is validated as a variance before anything is built from it, as the
python reference's `as_idio` does: a negative entry passed whenever the
contrast covariance stayed positive definite, so `D = [1, -0.5]` priced
0.9214 for a covariance with eigenvalue -0.5 (#367). The contrast is
formed from the factor grammar, `(V_j - V_k).(V_l - V_k) + D_k +
[j == l] D_j`, so a common loading cancels before it can round the
idiosyncratic variances away (#302), and the Cholesky ridge is relative
to the contrast scale: an absolute `1e-12` made the binary favourite
fall from 0.7602 to 0.5040 when the race was written in units 1e-8
(#409)."""
function ghk_choice_prob(mu::AbstractVector, V::AbstractMatrix, k::Int;
                         D = nothing, r_draws = 1000, seed = 9)
    J = length(mu)
    size(V, 1) == J || throw(ArgumentError(
        "V must have one row per alternative: got $(size(V, 1)) rows " *
        "for $J alternatives"))
    1 <= k <= J || throw(ArgumentError("k = $k is not an alternative of $J"))
    Dv = _check_D(D, J)                 # length J, or DimensionMismatch (#439)
    (all(isfinite, Dv) && all(isfinite, V) && all(isfinite, mu)) ||
        throw(ArgumentError("mu, V and D must be finite"))
    tol = 1e-12 * max(1.0, maximum(abs, Dv))
    for (j, x) in enumerate(Dv)
        x < -tol && throw(ArgumentError(
            "D[$j] = $x is negative: D is an idiosyncratic variance and " *
            "cannot be negative"))
    end
    Dv = max.(Dv, 0.0)                  # within tolerance: these ARE zero
    J == 1 && return 1.0
    m = J - 1
    others = [j for j in 1:J if j != k]
    a = [mu[j] - mu[k] for j in others]
    Vd = Float64.(V[others, :]) .- Float64.(V[k:k, :])
    C = Vd * Vd' .+ Dv[k] .+ [r == c ? Dv[others[r]] : 0.0
                              for r in 1:m, c in 1:m]
    ridge = 1e-12 * max(sum(C[d, d] for d in 1:m) / m, 1e-300)
    for d in 1:m
        C[d, d] += ridge
    end
    Lc = cholesky(Symmetric(C)).L
    rng = Xoshiro(seed)
    z = zeros(m, r_draws)
    logprob = zeros(r_draws)
    for t in 1:m
        mean_t = zeros(r_draws)
        for kk in 1:(t - 1)
            ltk = Lc[t, kk]
            ltk == 0 && continue
            mean_t .+= ltk .* z[kk, :]
        end
        ltt = Lc[t, t]
        for dr in 1:r_draws
            b = (-a[t] - mean_t[dr]) / ltt
            fb = ndtr(b)
            logprob[dr] += log(max(fb, TINY))
            u = rand(rng) * fb
            z[t, dr] = ndtri(clamp(u, TINY, 1 - 1e-16))
        end
    end
    return sum(exp, logprob) / r_draws
end

"""GHK simulated log-likelihood over all observations (CRN: the seed
per observation is a deterministic function of its index, so the
objective is smooth-in-parameters for a fixed draw set)."""
function ghk_loglik(mu::AbstractMatrix, V::AbstractMatrix,
                    choice::AbstractVector; r_draws = 1000, seed = 9)
    T = size(mu, 1)
    ll = 0.0
    for t in 1:T
        p = ghk_choice_prob(vec(mu[t, :]), V, Int(choice[t]);
                            r_draws = r_draws,
                            seed = seed + 7919 * t)
        ll += log(max(p, TINY))
    end
    return ll
end

# ---- the estimator ---------------------------------------------------

_fill_positions(J, r) = [(row, col) for col in 1:r for row in (col + 1):J]

"""Exact multinomial probit MLE for alternative-specific covariates.

    m = MNProbit(X, choice; intercepts = true)   # r = min(2, J - 2)
    fit!(m)                       # exact likelihood, analytic score
    fit!(m; method = :ghk)        # GHK simulated likelihood (CRN)
    predict_proba(m)

X: (T, J, p) covariates; choice: 1-based chosen alternatives.
`r` defaults to `min(2, J - 2)`; a rank above `J - 2` is refused because
the scale is then not identified (#201)."""
mutable struct MNProbit
    X::Array{Float64,3}
    choice::Vector{Int}
    T::Int
    J::Int
    p::Int
    r::Int
    pos::Vector{Tuple{Int,Int}}
    theta::Vector{Float64}
    beta::Vector{Float64}
    V::Matrix{Float64}
    loglik::Float64
    converged::Bool
    method::Symbol
    # Whether the constructor GENERATED the alternative intercept
    # columns. Without it, predict_proba could only guess from the
    # column count, and guessed wrong: see its docstring (#195).
    intercepts::Bool
    p_raw::Int          # covariate columns the caller supplied
end

"""Largest factor rank the unit-idiosyncratic parameterization
identifies at J alternatives. With D fixed to one and a zero reference
row, the free loadings number r*J - r*(r+1)/2 while the differenced
covariance has J*(J-1)/2 - 1 shape degrees of freedom; at r >= J-1 the
loading block is full rank, D = 1 no longer fixes the utility scale, and
(2 beta, V') prices every choice exactly like (beta, V) for a V' with
V'V'^T + I = 4 (V V^T + I) on contrasts. Coefficients and loadings then
move along an exact ridge (#201)."""
max_identified_rank(J::Integer) = max(J - 2, 0)

function MNProbit(X::AbstractArray{<:Real,3}, choice::AbstractVector;
                  intercepts = true, r = nothing)
    T, J, p0 = size(X)
    rmax = max_identified_rank(J)
    r = r === nothing ? min(2, rmax) : Int(r)
    (0 <= r <= rmax) || throw(ArgumentError(
        "factor rank r = " * string(r) * " is not identified at J = " *
        string(J) * " alternatives: with unit idiosyncratic variances " *
        "the utility scale is fixed only for r <= J - 2 = " *
        string(rmax) * " (#201)"))
    Xf = Float64.(X)
    if intercepts
        Z = zeros(T, J, J - 1)
        for j in 2:J
            Z[:, j, j - 1] .= 1.0
        end
        Xf = cat(Z, Xf; dims = 3)
    end
    p = size(Xf, 3)
    pos = _fill_positions(J, r)
    return MNProbit(Xf, _check_choice(choice, T, J), T, J, p, r, pos,
                    zeros(p + length(pos)), zeros(p), zeros(J, r),
                    NaN, false, :exact, intercepts, p0)
end

function _unpack(m::MNProbit, theta)
    beta = theta[1:m.p]
    V = zeros(eltype(theta), m.J, m.r)
    for (kk, (row, col)) in enumerate(m.pos)
        V[row, col] = theta[m.p + kk]
    end
    return beta, V
end

function _mu(m::MNProbit, beta)
    mu = zeros(eltype(beta), m.T, m.J)
    for pp in 1:m.p
        mu .+= m.X[:, :, pp] .* beta[pp]
    end
    return mu
end

function _nll_grad(m::MNProbit, theta)
    beta, V = _unpack(m, theta)
    mu = _mu(m, beta)
    ll, dmu, dV = choice_loglik_and_score(mu, V, m.choice)
    gbeta = [sum(dmu .* m.X[:, :, pp]) for pp in 1:m.p]
    gw = [dV[row, col] for (row, col) in m.pos]
    return -ll, -vcat(gbeta, gw)
end

"""Self-contained BFGS with backtracking (the winning house rule:
dependency-free; the analytic score makes this adequate)."""
function _bfgs(f_g, theta0; maxiter = 400, gtol = 1e-6)
    n = length(theta0)
    theta = copy(theta0)
    f, g = f_g(theta)
    H = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]
    for _ in 1:maxiter
        maximum(abs.(g)) < gtol && return theta, f, true
        d = -(H * g)
        step = 1.0
        f_new, g_new, theta_new = f, g, theta
        ok = false
        for _ in 1:40
            theta_new = theta .+ step .* d
            f_new, g_new = f_g(theta_new)
            if f_new <= f + 1e-4 * step * (g' * d)
                ok = true
                break
            end
            step *= 0.5
        end
        ok || return theta, f, false
        sdiff = theta_new .- theta
        y = g_new .- g
        sy = sdiff' * y
        if sy > 1e-12
            rho = 1 / sy
            A = [Float64(i == j) for i in 1:n, j in 1:n] .- rho .* (sdiff * y')
            H = A * H * A' .+ rho .* (sdiff * sdiff')
        end
        theta, f, g = theta_new, f_new, g_new
    end
    return theta, f, maximum(abs.(g)) < gtol
end

function fit!(m::MNProbit; method = :exact, maxiter = 400,
              r_draws = 1000, seed = 9)
    theta0 = vcat(zeros(m.p), fill(0.1, length(m.pos)))
    if method == :exact
        theta, nll, conv = _bfgs(t -> _nll_grad(m, t), theta0;
                                 maxiter = maxiter)
    elseif method == :ghk
        # CRN simulated likelihood, central-difference gradient (the
        # incumbent's own recipe; deterministic given the seed)
        function nll_ghk(t)
            beta, V = _unpack(m, t)
            return -ghk_loglik(_mu(m, beta), V, m.choice;
                               r_draws = r_draws, seed = seed)
        end
        function fg(t)
            f0 = nll_ghk(t)
            g = similar(t)
            h = 1e-4
            for i in eachindex(t)
                tp = copy(t); tp[i] += h
                tm = copy(t); tm[i] -= h
                g[i] = (nll_ghk(tp) - nll_ghk(tm)) / (2h)
            end
            return f0, g
        end
        theta, nll, conv = _bfgs(fg, theta0; maxiter = maxiter, gtol = 1e-4)
    else
        error("method must be :exact or :ghk")
    end
    m.theta = theta
    m.beta, m.V = _unpack(m, theta)
    m.loglik = -nll
    m.converged = conv
    m.method = method
    return m
end

loglikelihood(m::MNProbit) = m.loglik
coef(m::MNProbit) = copy(m.theta)

# Inference is for the EXACT likelihood only. After fit!(m; method=:ghk)
# the stored theta and loglik come from the CRN simulated likelihood, but
# every quantity below differentiates the exact-product objective, so the
# score, Hessian, covariance and printed standard errors described a
# DIFFERENT objective from the one fitted (#214). Refuse rather than
# mislabel; GHK inference would need the same CRN draws differentiated.
function _require_exact(m::MNProbit, what::AbstractString)
    m.method == :exact || throw(ArgumentError(
        what * " is the exact-likelihood quantity, but this model was " *
        "fitted with method = :" * string(m.method) * "; its theta " *
        "optimises a different objective. Refit with fit!(m) " *
        "(method = :exact) for inference"))
    return nothing
end

"""Per-observation score matrix G (T x nparams) at theta, from the
analytic gradients. Exact-likelihood fits only (#214)."""
function score_matrix(m::MNProbit, theta = m.theta)
    _require_exact(m, "score_matrix")
    beta, V = _unpack(m, theta)
    mu = _mu(m, beta)
    _, dmu, _, GV = choice_loglik_and_score(mu, V, m.choice;
                                            per_obs = true)
    np = m.p + length(m.pos)
    G = zeros(m.T, np)
    for pp in 1:m.p
        G[:, pp] .= vec(sum(dmu .* m.X[:, :, pp], dims = (2, 3)))
    end
    for (kk, (row, col)) in enumerate(m.pos)
        G[:, m.p + kk] .= GV[:, row, col]
    end
    return G
end

# armed by the ForwardDiff package extension: exact dual-mode
# derivative of the analytic score (forward-over-analytic)
const HESSIAN_ENGINE = Ref{Union{Nothing,Function}}(nothing)

"""Hessian of the log-likelihood at theta: central differences of the
ANALYTIC score by default (the score is exact, so this is ~1e-8);
loading ForwardDiff arms a machine-precision dual-mode engine.
Exact-likelihood fits only (#214)."""
function loglik_hessian(m::MNProbit, theta = m.theta; h = 1e-5)
    _require_exact(m, "loglik_hessian")
    eng = HESSIAN_ENGINE[]
    eng !== nothing && return eng(m, theta)
    np = length(theta)
    Hs = zeros(np, np)
    for i in 1:np
        tp = copy(theta); tp[i] += h
        tm = copy(theta); tm[i] -= h
        gp = -_nll_grad(m, tp)[2]
        gm = -_nll_grad(m, tm)[2]
        Hs[:, i] .= (gp .- gm) ./ (2h)
    end
    return (Hs .+ Hs') ./ 2
end

"""Parameter covariance at the fit: method = :hessian (observed
information, default), :opg (outer product of per-observation
scores), or :sandwich (H^-1 B H^-1, robust).

`:opg` never touches the Hessian. It used to compute and invert the
observed information first and only then branch, so a singular or
failing Hessian blocked the OPG fallback that exists for exactly that
case, and paid 2 * nparams score evaluations for nothing (#434)."""
function vcov(m::MNProbit; method = :hessian)
    method in (:hessian, :opg, :sandwich) ||
        error("method must be :hessian, :opg or :sandwich")
    _require_exact(m, "vcov")
    if method == :opg
        G = score_matrix(m)
        return inv(G' * G)
    end
    Hinv = inv(-loglik_hessian(m))
    method == :hessian && return (Hinv .+ Hinv') ./ 2
    G = score_matrix(m)
    return Hinv * (G' * G) * Hinv
end

stderror(m::MNProbit; method = :hessian) =
    sqrt.(max.(LinearAlgebra.diag(vcov(m; method = method)), 0.0))

function Base.show(io::IO, ::MIME"text/plain", m::MNProbit)
    println(io, "MNProbit  J=", m.J, " T=", m.T, " r=", m.r,
            "  method=", m.method,
            "  logLik=", round(m.loglik, digits = 3),
            m.converged ? "" : "  (NOT converged)")
    isnan(m.loglik) && return
    # standard errors exist only for the exact objective (#214)
    exact = m.method == :exact
    se = !exact ? fill(NaN, length(m.theta)) : try
        stderror(m)
    catch
        fill(NaN, length(m.theta))
    end
    names = vcat(["beta[$i]" for i in 1:m.p],
                 ["v[$row,$col]" for (row, col) in m.pos])
    println(io, rpad("param", 12), rpad("estimate", 12),
            exact ? "se" : "se (not available for a :" *
                           string(m.method) * " fit)")
    for i in eachindex(m.theta)
        println(io, rpad(names[i], 12),
                rpad(string(round(m.theta[i], digits = 4)), 12),
                exact ? round(se[i], digits = 4) : "-")
    end
end

"""Choice probabilities under the fitted parameters, by the same
factor-conditional product integrals (normalized across alternatives).

New data must carry the design the model was FITTED on. This used to decide whether to prepend generated intercept columns from
the column count alone -- `size(Xr, 3) != m.p` -- and the model recorded
neither whether it had generated them nor how many covariates the caller
supplied. So new data with the wrong number of features were silently
REINTERPRETED rather than refused: a model fitted on two covariates
without intercepts, handed one covariate, prepended two synthetic
intercept columns, then used only the first `m.p` of the three. It
applied coefficients fitted to the covariates to the intercepts and
IGNORED the supplied feature entirely, returning a plausible probability
row (#195).

The model now records both, so the column count is checked rather than
guessed. `p_raw` columns means raw covariates, and the intercepts are
generated exactly when the fit generated them; `p` columns means the
design is already assembled. Anything else raises."""
function predict_proba(m::MNProbit; X = nothing)
    Xf = X === nothing ? m.X : begin
        Xr = Float64.(X)
        size(Xr, 2) == m.J || throw(DimensionMismatch(
            "X has " * string(size(Xr, 2)) * " alternatives; the model " *
            "was fitted on " * string(m.J)))
        nc = size(Xr, 3)
        if nc == m.p
            # already the fitted design, intercepts and all
        elseif m.intercepts && nc == m.p_raw
            T2 = size(Xr, 1)
            Z = zeros(T2, m.J, m.J - 1)
            for j in 2:m.J
                Z[:, j, j - 1] .= 1.0
            end
            Xr = cat(Z, Xr; dims = 3)
        else
            throw(DimensionMismatch(
                "X has " * string(nc) * " covariate columns; this model " *
                "was fitted on " * string(m.p_raw) *
                (m.intercepts ? " covariates plus generated intercepts, so " *
                 "pass either " * string(m.p_raw) * " or " * string(m.p) :
                 " covariates and no generated intercepts, so pass " *
                 string(m.p))))
        end
        Xr
    end
    T2 = size(Xf, 1)
    mu = zeros(T2, m.J)
    for pp in 1:m.p
        mu .+= Xf[:, :, pp] .* m.beta[pp]
    end
    V = m.V .- sum(m.V, dims = 1) ./ m.J
    sharp = sqrt(2) * maximum(sqrt.(vec(sum(V .^ 2, dims = 2))))
    F, W = nodes_for_likelihood(m.r; sharp = sharp)
    Fq = F[:, 1:m.r]
    zq = F[:, m.r + 1]
    Vf = Fq * V'
    P = zeros(T2, m.J)
    for k in 1:m.J
        acc = zeros(T2, length(W))
        for j in 1:m.J
            j == k && continue
            shift = Vf[:, k] .- Vf[:, j] .+ zq
            acc .+= log.(max.(ndtr.((mu[:, k] .- mu[:, j]) .+ shift'), TINY))
        end
        mx = vec(maximum(acc, dims = 2))
        P[:, k] = exp.(mx) .* (exp.(acc .- mx) * W)
    end
    return P ./ sum(P, dims = 2)
end

end # module
