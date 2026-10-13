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
using LinearAlgebra: SymTridiagonal, eigen, cholesky, Symmetric, issuccess, eigmin
using Random: Xoshiro

export MNProbit, max_identified_rank, fit!, loglikelihood, predict_proba, coef, vcov,
    mass_defect,
    stderror, score_matrix, loglik_hessian,
    choice_loglik_and_score, ghk_choice_prob, ghk_choice_logprob

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

# the tail of the erfcx continued fraction, erfcx(x) = 1/(sqrt(pi)(x + cf)),
# from n terms: 60 is converged from x = 2.5, 12 from x = 14 (the n-term
# truncation error there is below 1e-19 relative)
function _erfcx_cf(x::Real, n::Int = 60)
    cf = 0.0
    for k in n:-1:1
        cf = (k / 2) / (x + cf)
    end
    return cf
end

_erfcx(x::Real) = 1 / (sqrt(pi) * (x + _erfcx_cf(x)))

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
# Cephes ndtr is RELATIVELY accurate in the left tail until it underflows
# near -37.5 (erfc is exp(-x^2) times a rational function), so log(ndtr)
# is exact enough down to -20, scipy's log_ndtr crossover. The 60-term
# continued fraction is kept for the far tail only: taking it from
# -3.54 put ~60 divisions on every tail node and doubled the Fishing
# fit (12.7 s -> 24.5 s, the same optimum to 1e-13; #716).
const _LOGNDTR_TAIL = -20.0

function logndtr(z::Real)
    # cephes ndtr in the ordinary range, not _ndtr_series: the series
    # runs up to 119 terms and this is called on every node of every
    # Newton step of every likelihood evaluation, which made an
    # end-to-end fit so slow it looked like a hang. Above the cut the
    # CDF is O(1e-4) or larger, so log of it loses nothing.
    if z > _LOGNDTR_CUT
        return log1p(-ndtr(-z))                  # the complement is tiny
    elseif z >= _LOGNDTR_TAIL
        return log(ndtr(z))
    end
    x = -z / sqrt(2)                             # x >= 14: 12 terms suffice
    return log(0.5 / (sqrt(pi) * (x + _erfcx_cf(x, 12)))) - 0.5 * z * z
end

"""log of phi(a)/Phi(a), the inverse Mills ratio, stable in the far tail."""
_log_mills(a::Real) = -0.5 * a * a - 0.5 * log(2 * pi) - logndtr(a)

"""(lam, a + lam) for the inverse Mills ratio lam = phi(a)/Phi(a).

lam (a + lam) is the tilt's curvature term and lies in (0, 1), but in
the far left tail lam ~ -a and the sum cancels: through
exp(-a^2/2 - log(2pi)/2 - logndtr(a)) lam carried the rounding of a^2/2
(2e-7 relative at a = -5.7e4), the sum came out negative, the tilt
curvature h = -196 and sqrt(h) threw a DomainError in the Fishing fit
(#716). Below _LOGNDTR_TAIL the continued fraction gives both
without cancellation: lam = sqrt(2)(x + cf) = -a + sqrt(2) cf,
a + lam = sqrt(2) cf, x = -a/sqrt(2). Port of
winning/likelihood.py::_mills_gap."""
function _mills_gap(a::Real)
    if a < _LOGNDTR_TAIL
        g = sqrt(2) * _erfcx_cf(-a / sqrt(2), 12)
        return -a + g, g
    end
    lam = exp(-0.5 * a * a - 0.5 * log(2 * pi) - logndtr(a))
    return lam, a + lam
end

# ---- node rules -------------------------------------------------------

"""All permutations of 1:r (lexicographic)."""
function _permutations(r::Int)
    r == 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    for p in _permutations(r - 1)
        for i in 0:(r - 1)
            q = copy(p)
            insert!(q, i + 1, r)
            push!(out, q)
        end
    end
    return sort(out)
end

# Factor orientation is arbitrary: V and V*Q price the same law for any
# orthogonal Q, and in particular for a column sign flip or permutation.
# The Gauss-Hermite tensor is invariant under both; a Halton cloud is
# not, so the sharp-branch probability moved by 0.0029 when one loading
# column changed sign (#605). The cloud is the orbit of base points under
# the signed permutations of the factor coordinates (all 2^r r! of them
# while that is at most 384, i.e. r <= 4; the 2^r sign flips beyond), at
# about the same node count. General rotations are not covered.
const _MAX_SYMMETRY_ORDER = 384

function _factor_symmetries(r::Int, n::Int)
    r == 0 && return [Int[]], [Float64[]]
    signs = [Float64[((s >> (c - 1)) & 1) == 0 ? 1.0 : -1.0 for c in 1:r]
             for s in 0:(2^r - 1)]
    full = 2^r * factorial(r)
    if full <= min(_MAX_SYMMETRY_ORDER, n)
        return _permutations(r), signs
    elseif 2^r <= n
        return [collect(1:r)], signs
    end
    return [collect(1:r)], [ones(r)]
end

"""Halton normal cloud of about `n` points over (factor^r, extra) whose
factor coordinates are closed under the signed permutations of
`_factor_symmetries` (#605). Equal weights."""
function _symmetric_halton(r::Int, extra::Int, n::Int)
    perms, signs = _factor_symmetries(r, n)
    order = length(perms) * length(signs)
    nb = cld(n, order)
    X, _ = _halton_normal(r + extra, nb)
    F = zeros(nb * order, r + extra)
    row = 0
    for pm in perms, sg in signs
        for b in 1:nb
            row += 1
            for c in 1:r
                F[row, c] = sg[c] * X[b, pm[c]]
            end
            for c in (r + 1):(r + extra)
                F[row, c] = X[b, c]
            end
        end
    end
    return F, fill(1.0 / size(F, 1), size(F, 1))
end

"""(F, W): nodes over (factor^r, own-noise); a symmetrised Halton cloud
past sharpness 3 (the dependency-free escalation of the R port, #605)."""
function nodes_for_likelihood(r::Int; Qf = 7, Qz = 7, sharp = 0.0)
    sharp > 3.0 && return _symmetric_halton(r, 1, 2^10)
    xf, wf = _gh1(Qf)
    xz, wz = _gh1(Qz)
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
        string(length(choice)) * " for " * string(T) * " observations"))
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
the likelihood (#439). A scalar is the same variance for every
alternative, the python `as_idio` shorthand; `collect` cannot iterate a
`Float64`, so `D = 1.0` failed in both exported engines (#503). Non-
finite entries are refused here; the sign contract belongs to each
engine (the likelihood needs D > 0, GHK admits a zero)."""
function _check_D(D, J::Integer)
    D === nothing && return ones(J)
    Dv = D isa Number ? fill(Float64(D), J) : Float64.(vec(collect(D)))
    length(Dv) == J || throw(DimensionMismatch(
        "D must have one idiosyncratic variance per alternative: got " *
        string(length(Dv)) * " for J = " * string(J)))
    all(isfinite, Dv) || throw(ArgumentError(
        "D must be finite: it is an idiosyncratic variance"))
    return Dv
end

"""A caller's (F, W) rule for a rank-`r` likelihood: exactly `r + 1`
columns (r factor coordinates, then own noise), one finite row per
weight. Columns were sliced by position, so a surplus column was read
as the own-noise coordinate and the real one dropped, moving a 0.5
probability to 1.0 without an error (#476).

The weights are RELATIVE, as every other port's quadrature boundary
treats them: finite, non-negative, positive total, and normalized here
(by the maximum first, so a finite 1e308 rule does not overflow). They
were used as given, so `W = [5, 5]` added log(10) per observation, a
signed rule returned a log-likelihood above zero and an all-zero rule a
NaN score (#263)."""
function _check_nodes(nodes, r::Integer)
    F, W = nodes
    F isa AbstractMatrix || throw(ArgumentError(
        "nodes F must be a matrix with r + 1 = " * string(r + 1) *
        " columns"))
    size(F, 2) == r + 1 || throw(DimensionMismatch(
        "nodes F must have r + 1 = " * string(r + 1) * " columns (r " *
        "factor coordinates, then own noise); got " * string(size(F, 2))))
    W isa AbstractVector || throw(ArgumentError("nodes W must be a vector"))
    (size(F, 1) == length(W) && !isempty(W)) || throw(DimensionMismatch(
        "nodes need one weight per row of F: got " * string(length(W)) *
        " weights for " * string(size(F, 1)) * " nodes"))
    all(isfinite, F) || throw(ArgumentError("nodes F has a non-finite entry"))
    Wv = Float64.(collect(W))
    for (i, w) in enumerate(Wv)
        isfinite(w) || throw(ArgumentError(
            "nodes W[" * string(i) * "] = " * string(w) * " is not finite"))
        w >= 0 || throw(ArgumentError(
            "nodes W[" * string(i) * "] = " * string(w) * " is negative: " *
            "quadrature weights are relative masses (#263)"))
    end
    mx = maximum(Wv)
    mx > 0 || throw(ArgumentError(
        "nodes W are all zero: a quadrature rule needs positive total " *
        "weight (#263)"))
    Wv = Wv ./ mx
    return F, Wv ./ sum(Wv)
end

# ---- the sharpness blend (#420) ----------------------------------------

# The node family switches from the Hermite tensor to the Halton cloud at
# sharpness 3, and the two rules disagree there: at the binary #420
# fixture the hard switch moved the log-likelihood by 0.0243 for a 1e-10
# change in a loading. Across this window the likelihood is the
# probability-weighted blend (1 - w) P_GH + w P_Halton with a C1
# smoothstep w, and the score carries the blend's own derivative.
const _BLEND_LO, _BLEND_HI = 2.9, 3.1

function _blend_weight(sharp::Real)
    t = (sharp - _BLEND_LO) / (_BLEND_HI - _BLEND_LO)
    t <= 0 && return 0.0, 0.0
    t >= 1 && return 1.0, 0.0
    return t * t * (3 - 2t), 6t * (1 - t) / (_BLEND_HI - _BLEND_LO)
end

# The blend's statistic must be differentiable and symmetric in the
# alternatives: a max over row norms has a cusp at ties. A p-norm of the
# centred row norms is smooth, permutation symmetric, at least the max
# and at most J^(1/p) above it (the python reference's _smooth_sharpness).
const _SHARP_P = 32.0

function _smooth_sharpness(V::AbstractMatrix, Dv::AbstractVector)
    J = size(V, 1)
    C = V .- sum(V, dims = 1) ./ J
    n2 = vec(sum(C .^ 2, dims = 2))
    M = maximum(n2)
    sc = sqrt(2) / sqrt(minimum(Dv))
    M > 0 || return zero(eltype(n2)), zeros(eltype(V), size(V))
    u = n2 ./ M
    up = u .^ (_SHARP_P / 2)
    Su = sum(up)
    sn = sqrt(M) * Su^(1 / _SHARP_P)
    dn2 = 0.5 .* sn .* (u .^ (_SHARP_P / 2 - 1)) ./ (M * Su)
    G = 2 .* dn2 .* C
    G = G .- sum(G, dims = 1) ./ J
    return sc * sn, sc .* G
end

# ---- the own-noise tilt (#128, #270, #352) -----------------------------

# The nodes sample the chosen alternative's own noise z where its PRIOR
# has mass, but when the chosen alternative is behind, the integrand
# h(z) = phi(z) prod_j Phi((c_j + s_k z)/s_j) has its mass far to the
# right (at a gap of -20 the binary likelihood was -143.2 against
# -103.6). On the Gauss-Hermite tensor each observation's own-noise nodes
# move to the mode m of log h and scale with the curvature there,
# z = m + sigma x, and the change of measure rides in the weights,
# log sigma - z^2/2 + x^2/2. The mode is taken with the factors
# integrated out in the Gaussian approximation (rival noise widened to
# t_j^2 = s_j^2 + |v_k - v_j|^2). m is a FIXED smooth map -- four Newton
# steps on a Mills-ratio surrogate, then two exact ones -- whose
# derivatives are carried along, so the analytic score is the exact
# gradient of the returned value (an earlier, converged-Newton tilt with
# a frozen shift made fits ten times slower and stop converging). Port
# of winning/likelihood.py::_tilt_mode; the Halton branch is not tilted.
const _TILT_SURROGATE_STEPS, _TILT_EXACT_STEPS = 4, 2

"""(m, h, dm_c, dh_c, dm_t, dh_t) for one observation: the tilt centre,
the curvature at the last exact point, and their derivatives in the
contrasts c_j and the effective rival scales t_j."""
function _tilt_mode(c0::AbstractVector, t::AbstractVector, sk::Real)
    R = length(c0)
    ET = promote_type(eltype(c0), eltype(t), typeof(sk))
    m = zero(ET)
    dm_c = zeros(ET, R)
    dm_t = zeros(ET, R)
    dh_c = zeros(ET, R)
    dh_t = zeros(ET, R)
    h = one(ET)
    lam = zeros(ET, R); kap = zeros(ET, R); kp = zeros(ET, R)
    a = zeros(ET, R)
    for it in 1:(_TILT_SURROGATE_STEPS + _TILT_EXACT_STEPS)
        for j in 1:R
            a[j] = (c0[j] + sk * m) / t[j]
            if it <= _TILT_SURROGATE_STEPS
                q = sqrt(a[j]^2 + 4)
                lam[j] = (q - a[j]) / 2
                kap[j] = (1 - a[j] / q) / 2
                kp[j] = -2 / q^3
            else
                lam[j], gap = _mills_gap(a[j])
                kap[j] = lam[j] * gap
                kp[j] = lam[j] * (1 - kap[j]) - kap[j] * gap
            end
        end
        g1 = -m
        h = one(ET)
        S3 = zero(ET)
        for j in 1:R
            rho = sk / t[j]
            g1 += rho * lam[j]
            h += rho^2 * kap[j]
            S3 += rho^3 * kp[j]
        end
        sc = g1 / h^2
        for j in 1:R
            rho = sk / t[j]
            dg1c = -(rho * kap[j]) / t[j] - h * dm_c[j]
            dh_c[j] = rho^2 * kp[j] / t[j] + S3 * dm_c[j]
            dg1t = rho * (kap[j] * a[j] - lam[j]) / t[j] - h * dm_t[j]
            dh_t[j] = -rho^2 * (2 * kap[j] + kp[j] * a[j]) / t[j] +
                      S3 * dm_t[j]
            dm_c[j] += dg1c / h - sc * dh_c[j]
            dm_t[j] += dg1t / h - sc * dh_t[j]
        end
        m += g1 / h
    end
    return m, h, dm_c, dh_c, dm_t, dh_t
end

"""Per-observation log P(choice), d/dmu and d/dV (T, J, r) on one node
set; V gauge-centred; `tilt` moves each observation's own-noise nodes
(Gauss-Hermite branch only). Port of winning/likelihood.py::_choice_terms."""
function _choice_terms(mu::AbstractMatrix, V::AbstractMatrix,
                       choice::AbstractVector{Int}, s::AbstractVector,
                       F::AbstractMatrix, W::AbstractVector; tilt = true)
    T, J = size(mu)
    r = size(V, 2)
    ET = promote_type(eltype(mu), eltype(V), Float64)
    Q = length(W)
    Fq = F[:, 1:r]
    xq = F[:, r + 1]
    logW = log.(W)
    Vf = Fq * V'                                # (Q, J)
    lp = zeros(ET, T)
    dmu = zeros(ET, T, J)
    GV = zeros(ET, T, J, r)
    R = J - 1
    logPhi = zeros(ET, R, Q)
    lam = zeros(ET, R, Q)
    acc = zeros(ET, Q)
    z = zeros(ET, Q)
    c0 = zeros(ET, R)
    tj = zeros(ET, R)
    rv = zeros(Int, R)
    dcs = zeros(ET, R)
    Hj = zeros(ET, R, r)
    for t in 1:T
        k = choice[t]
        i = 0
        for j in 1:J
            j == k && continue
            i += 1
            rv[i] = j
        end
        for (i, j) in enumerate(rv)
            c0[i] = mu[t, k] - mu[t, j]
            tj[i] = sqrt(s[j]^2 + sum((V[k, c] - V[j, c])^2 for c in 1:r;
                                      init = zero(ET)))
        end
        if tilt
            m, h, dm_c, dh_c, dm_t, dh_t = _tilt_mode(c0, tj, s[k])
            sig = 1 / sqrt(h)
        else
            m, sig = zero(ET), one(ET)
        end
        @inbounds for q in 1:Q
            z[q] = m + sig * xq[q]
            a_acc = logW[q]
            for (i, j) in enumerate(rv)
                A = (c0[i] + Vf[q, k] - Vf[q, j] + s[k] * z[q]) / s[j]
                lP = logndtr(A)
                logPhi[i, q] = lP
                lam[i, q] = exp(-A^2 / 2 - 0.5 * log(2 * pi) - lP)
                a_acc += lP
            end
            tilt && (a_acc += xq[q]^2 / 2 - z[q]^2 / 2)
            acc[q] = a_acc
        end
        mx = maximum(acc)
        rs = zero(ET)
        for q in 1:Q
            acc[q] = exp(acc[q] - mx)
            rs += acc[q]
        end
        lp[t] = log(sig) + mx + log(max(rs, TINY))
        Eg = zero(ET); Egx = zero(ET)
        fill!(dcs, zero(ET))
        fill!(Hj, zero(ET))
        # the factor score is accumulated per rival and written to GV
        # once per observation: scattering it into the (T, J, r) array
        # on every node cost more than the likelihood itself
        @inbounds for q in 1:Q
            om = acc[q] / rs
            gp = -z[q]
            for (i, j) in enumerate(rv)
                wl = om * lam[i, q] / s[j]
                dcs[i] += wl
                for c in 1:r
                    Hj[i, c] += wl * Fq[q, c]
                end
                gp += s[k] / s[j] * lam[i, q]
            end
            Eg += om * gp
            Egx += om * gp * xq[q]
        end
        for (i, j) in enumerate(rv), c in 1:r
            GV[t, k, c] += Hj[i, c]
            GV[t, j, c] -= Hj[i, c]
        end
        for (i, j) in enumerate(rv)
            dci = dcs[i]
            if tilt
                dsc = -0.5 * sig / h * dh_c[i]
                dst = -0.5 * sig / h * dh_t[i]
                dci += dsc / sig + Eg * dm_c[i] + Egx * dsc
                dt = dst / sig + Eg * dm_t[i] + Egx * dst
                # t_j = sqrt(s_j^2 + |v_k - v_j|^2)
                for c in 1:r
                    gvc = dt / tj[i] * (V[k, c] - V[j, c])
                    GV[t, k, c] += gvc
                    GV[t, j, c] -= gvc
                end
            end
            dmu[t, k] += dci
            dmu[t, j] -= dci
        end
    end
    return lp, dmu, GV
end

"""J = 2 in closed form: P(k) = Phi(c/s), c = mu_k - mu_j,
s^2 = D_k + D_j + |v_k - v_j|^2, with its analytic score. No quadrature,
so no tail, no dependence on a null loading column (#345)."""
function _choice_pair(mu::AbstractMatrix, V::AbstractMatrix,
                      choice::AbstractVector{Int}, Dv::AbstractVector)
    T = size(mu, 1)
    r = size(V, 2)
    ET = promote_type(eltype(mu), eltype(V), Float64)
    lp = zeros(ET, T)
    dmu = zeros(ET, T, 2)
    GV = zeros(ET, T, 2, r)
    for t in 1:T
        k = choice[t]
        j = 3 - k
        c = mu[t, k] - mu[t, j]
        s2 = Dv[k] + Dv[j] + sum((V[k, cc] - V[j, cc])^2 for cc in 1:r;
                                 init = zero(ET))
        if s2 > 0
            s = sqrt(s2)
            zz = c / s
            l = logndtr(zz)
            lam = exp(-zz^2 / 2 - 0.5 * log(2 * pi) - l)
            lp[t] = l
            dmu[t, k] = lam / s
            dmu[t, j] = -lam / s
            for cc in 1:r
                g = -lam * c / s^3 * (V[k, cc] - V[j, cc])
                GV[t, k, cc] = g
                GV[t, j, cc] = -g
            end
        else
            lp[t] = c > 0 ? zero(ET) : (c < 0 ? ET(-Inf) : log(ET(0.5)))
        end
    end
    return lp, dmu, GV
end

# Own-noise Gauss-Hermite order when the caller gives none: 7 below
# J = 6, 11 from J = 6, where the product of J - 1 rival CDFs is sharper
# than seven nodes resolve (python: winning.likelihood.default_qz, #173).
default_qz(J::Integer) = J >= 6 ? 11 : 7

"""Per-observation log P(choice_t) with score rows (lp, dmu, GV): the
one probability map that fitting, inference and prediction share."""
function _choice_logprob_terms(mu::AbstractMatrix, V::AbstractMatrix,
                               choice::Vector{Int}, Dv::Vector{Float64};
                               Qf = 7, Qz = nothing, nodes = nothing)
    T, J = size(mu)
    r = size(V, 2)
    Qz === nothing && (Qz = default_qz(J))
    if nodes === nothing && J == 2
        # the closed form needs only D_k + D_j + |v_k - v_j|^2 > 0
        all(>=(0.0), Dv) || throw(ArgumentError(
            "D must be non-negative: it is an idiosyncratic variance"))
        return _choice_pair(mu, V, choice, Dv)
    end
    all(>(0.0), Dv) || throw(ArgumentError(
        "D must be strictly positive for the likelihood: the own-noise " *
        "integral divides by each rival's sqrt(D)"))
    s = sqrt.(Dv)
    Vc = V .- sum(V, dims = 1) ./ J          # gauge: differences decide
    if nodes !== nothing
        F, W = _check_nodes(nodes, r)
        # a caller's rule is integrated as given (no tilt)
        return _choice_terms(mu, Vc, choice, s, F, W; tilt = false)
    end
    # A loading column that is zero after centring changes no choice
    # probability, but it changed the node dimension, so a zero-padded V
    # priced differently from the unpadded one (#345). Integrate over
    # the live columns only; a null column's score is exactly zero.
    live = [c for c in 1:r if any(!=(0), Vc[:, c])]
    if length(live) < r
        lp, dmu, GVl = _choice_logprob_terms(mu, V[:, live], choice, Dv;
                                             Qf = Qf, Qz = Qz)
        GV = zeros(eltype(GVl), T, J, r)
        GV[:, :, live] .= GVl
        return lp, dmu, GV
    end
    # a rival much less noisy than the chosen alternative makes the
    # own-noise integrand a step of width s_j/s_k: more own-noise nodes,
    # capped at 4x (#352); D = 1 is unaffected
    ratio = sqrt(maximum(Dv) / minimum(Dv))
    Qz = ratio > 1 ? Int(min(4 * Qz, ceil(Qz * ratio))) : Qz
    sharp, sgrad = _smooth_sharpness(V, Dv)
    w, dw = _blend_weight(sharp)
    if w == 0.0 || w == 1.0
        F, W = nodes_for_likelihood(r; Qf = Qf, Qz = Qz, sharp = sharp)
        return _choice_terms(mu, Vc, choice, s, F, W; tilt = (w == 0.0))
    end
    Fg, Wg = nodes_for_likelihood(r; Qf = Qf, Qz = Qz, sharp = 0.0)
    Fs, Ws = nodes_for_likelihood(r; Qf = Qf, Qz = Qz, sharp = Inf)
    lg, dmg, GVg = _choice_terms(mu, Vc, choice, s, Fg, Wg; tilt = true)
    ls, dms, GVs = _choice_terms(mu, Vc, choice, s, Fs, Ws; tilt = false)
    lp = similar(lg)
    dmu = similar(dmg)
    GV = similar(GVg)
    for t in eachindex(lg)
        x1 = log1p(-w) + lg[t]
        x2 = log(w) + ls[t]
        mx = max(x1, x2)
        lp[t] = mx + log(exp(x1 - mx) + exp(x2 - mx))
        a = exp(x1 - lp[t])
        b = exp(x2 - lp[t])
        dmu[t, :] .= a .* dmg[t, :] .+ b .* dms[t, :]
        gw = (exp(ls[t] - lp[t]) - exp(lg[t] - lp[t])) * dw
        GV[t, :, :] .= a .* GVg[t, :, :] .+ b .* GVs[t, :, :] .+ gw .* sgrad
    end
    return lp, dmu, GV
end


# Lower Cholesky factor of a contrast covariance, AS GIVEN whenever it has
# one (python: winning.methods.native._contrast_cholesky). An absolute
# 1e-12 ridge was model noise in the utility unit (#409), and the relative
# ridge that replaced it, added always, still inflated a valid small
# pivot: near-twins with variance 2^-43 priced [0.5, 0.167, 0.333] for the
# exact [0.5, 0.079, 0.421]; its 1e-300 floor was an absolute ridge below
# that scale. The matrix is brought to unit scale by an exact power of
# two, factorised unmodified, given a ridge of 1e-12 of its unit mean
# variance only if that fails, and the factor scaled back exactly. An
# exactly zero contrast (a tie, Dirac scope) keeps the old floor.
function _contrast_cholesky(C::AbstractMatrix{Float64})
    m = size(C, 1)
    tr = sum(C[d, d] for d in 1:m) / m
    if !(tr > 0.0 && isfinite(tr))
        Cr = copy(C)
        for d in 1:m
            Cr[d, d] += 1e-12 * max(tr, 1e-300)
        end
        return Matrix(cholesky(Symmetric(Cr)).L)
    end
    k = -fld(exponent(tr) + 1, 2)
    Cs = ldexp.(C, 2k)
    F = cholesky(Symmetric(Cs); check = false)
    if !issuccess(F)
        ts = sum(Cs[d, d] for d in 1:m) / m
        for d in 1:m
            Cs[d, d] += 1e-12 * ts
        end
        F = cholesky(Symmetric(Cs))
    end
    return ldexp.(Matrix(F.L), -k)
end

"""Log-likelihood of observed argmax choices with the ANALYTIC score.
mu: (T, J) utilities; V: (J, r) loadings; choice: 1-based indices.
Returns (loglik, dmu, dV), plus the per-observation loading scores
(T, J, r) with `per_obs = true`. Port of winning/likelihood.py: J = 2
in closed form; J >= 3 integrates the factors and the chosen
alternative's own noise on a Gauss-Hermite tensor with tilted own-noise
nodes, or past sharpness 3 a symmetrised Halton cloud, blended
continuously across the switch. `Qz = nothing` takes 7 own-noise nodes
below J = 6 and 11 from J = 6 (#173). `nodes = (F, W)` integrates the
caller's rule as given."""
function choice_loglik_and_score(mu::AbstractMatrix, V::AbstractMatrix,
                                 choice::AbstractVector;
                                 D = nothing, Qf = 7, Qz = nothing,
                                 nodes = nothing, per_obs = false)
    T, J = size(mu)
    # Every observation must be accounted for: a row whose choice is
    # outside 1..J was never visited, and a SHORTER choice vector
    # dropped the tail without a word (#194).
    choice = _check_choice(choice, T, J)
    # one loading row per alternative, BEFORE gauge centring and the
    # sharpness dispatch (#462)
    size(V, 1) == J || throw(DimensionMismatch(
        "V must have one loading row per alternative: got " *
        string(size(V, 1)) * " rows for J = " * string(J)))
    Dv = _check_D(D, J)
    nodes === nothing || _check_nodes(nodes, size(V, 2))
    lp, dmu, GV = _choice_logprob_terms(mu, V, choice, Dv;
                                        Qf = Qf, Qz = Qz, nodes = nodes)
    loglik = sum(lp)
    dV = dropdims(sum(GV, dims = 1), dims = 1)
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
    return exp(ghk_choice_logprob(mu, V, k; D = D, r_draws = r_draws,
                                  seed = seed))
end

"""log P(alternative k chosen | mu row) by the same GHK sampler as
`ghk_choice_prob`, kept in log space throughout (#270): each draw's
conditional factors are summed as `logndtr`, the truncated draw is
inverted from its log CDF (`_ndtri_log`), and the draws are averaged by
log-sum-exp. The old route floored every factor and every truncated
uniform at 1e-300, so the log-likelihood went flat at log(1e-300) =
-690.78 once the chosen alternative trailed by about 50 utility units,
with a zero score, while the exact value kept falling (-986 at -55)."""
function ghk_choice_logprob(mu::AbstractVector, V::AbstractMatrix, k::Int;
                            D = nothing, r_draws = 1000, seed = 9)
    _check_r_draws(r_draws)
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
    J == 1 && return 0.0
    m = J - 1
    others = [j for j in 1:J if j != k]
    a = [mu[j] - mu[k] for j in others]
    Vd = Float64.(V[others, :]) .- Float64.(V[k:k, :])
    C = Vd * Vd' .+ Dv[k] .+ [r == c ? Dv[others[r]] : 0.0
                              for r in 1:m, c in 1:m]
    Lc = _contrast_cholesky(C)
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
            lfb = logndtr(b)
            logprob[dr] += lfb
            # the truncated draw Phi^-1(u Phi(b)), from its log so a deep
            # bound keeps z near b instead of pinning it at Phi^-1(1e-300)
            lu = log(max(rand(rng), floatmin(Float64))) + lfb
            z[t, dr] = _ndtri_log(min(lu, log1p(-1e-16)))
        end
    end
    top = maximum(logprob)
    isfinite(top) || return top
    return top + log(sum(x -> exp(x - top), logprob)) - log(r_draws)
end

"""Phi^-1(exp(lp)) for lp <= 0, accurate where exp(lp) underflows or
falls below ndtri's range: ndtri itself while exp(lp) is an ordinary
double, else Newton on logndtr(x) = lp from the tail asymptote (logndtr
is concave, so the iteration is monotone)."""
function _ndtri_log(lp::Float64)
    lp > -600.0 && return ndtri(exp(lp))
    t = -2 * lp
    x = -sqrt(t - log(t) - log(2 * pi))
    for _ in 1:30
        g = logndtr(x) - lp
        d = exp(-0.5 * x * x - 0.5 * log(2 * pi) - logndtr(x))
        step = g / d
        x -= step
        abs(step) <= 1e-15 * abs(x) && break
    end
    return x
end

# A draw budget is a positive whole number: r_draws = 0 averaged an
# empty set, 0/0 = NaN, and fit!(method = :ghk) carried it into a NaN
# log-likelihood. The rust/fastrace boundary already refused it (#478).
function _check_r_draws(r_draws)
    (r_draws isa Integer && r_draws >= 1) || throw(ArgumentError(
        "r_draws must be at least 1; got " * string(r_draws)))
    return nothing
end

"""GHK simulated log-likelihood over all observations, with common
random numbers: every observation uses the SAME base-uniform cloud (the
stream of `seed`), so the objective is smooth in the parameters for a
fixed draw set AND a function of the sample, not of its row order. The
stream used to be `seed + 7919 t`, keyed by row position, so swapping two
rows reassigned their draws and moved the log-likelihood by 1.18 nats
on a two-row sample (#619)."""
function ghk_loglik(mu::AbstractMatrix, V::AbstractMatrix,
                    choice::AbstractVector; r_draws = 1000, seed = 9)
    _check_r_draws(r_draws)
    T = size(mu, 1)
    ll = 0.0
    for t in 1:T
        ll += ghk_choice_logprob(vec(mu[t, :]), V, Int(choice[t]);
                                 r_draws = r_draws, seed = seed)
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
    # largest |1 - sum_k raw P_k| on the training rows: how far the raw
    # per-choice masses the fit maximizes are from the normalized
    # predict_proba (#376); NaN before an exact fit
    mass_defect::Float64
    # a recession direction of the mean design (no finite MLE), or empty
    separation::Vector{Float64}
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

"""A finite, non-negative whole number. `Int(1.9)` threw an
InexactError rather than an argument error (#479), and an unchecked
`maxiter` went straight into a range (#560)."""
function _as_count(x, name::AbstractString)
    ok = !(x isa Bool) && x isa Real && isfinite(x) && isinteger(x) && x >= 0
    ok || throw(ArgumentError(name * " must be a finite non-negative " *
                              "whole number; got " * repr(x)))
    return Int(x)
end

function MNProbit(X::AbstractArray{<:Real,3}, choice::AbstractVector;
                  intercepts = true, r = nothing)
    T, J, p0 = size(X)
    # An empty design has an empty-sum likelihood with zero score, so
    # BFGS stopped at once and reported the initial coefficients as a
    # converged fit (#591).
    T > 0 || throw(ArgumentError("at least one observation is required"))
    # One alternative is no choice at all: every probability is one and
    # the objective is flat, so the fit returned its initializer as a
    # converged estimate (#581).
    J >= 2 || throw(ArgumentError(
        "a choice model needs at least two alternatives; X has " *
        string(J) * ". With one, every coefficient is unidentified (#581)"))
    rmax = max_identified_rank(J)
    r = r === nothing ? min(2, rmax) : _as_count(r, "factor rank r")
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
                    NaN, false, :exact, intercepts, p0, NaN, Float64[])
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

# A stationary point to double precision. Near the optimum of a
# likelihood summed over hundreds of observations the Newton decrement
# g' H g falls below the resolution of f (eps |f|) while the score is
# still a little above gtol: no step can then be confirmed by the
# objective, and BFGS spun to maxiter (or failed its line search) at a
# score of 3e-6 -- converged or not depending on summation order, i.e.
# on whether bounds checking was on. Within 100 gtol, that is a stop.
function _precision_stop(f, g, H; gtol = 1e-6)
    isempty(g) && return true
    maximum(abs.(g)) < 100 * gtol || return false
    return abs(g' * (H * g)) <= 64 * eps(Float64) * max(1.0, abs(f))
end

"""Self-contained BFGS with backtracking (the winning house rule:
dependency-free; the analytic score makes this adequate)."""
function _bfgs(f_g, theta0; maxiter = 400, gtol = 1e-6)
    n = length(theta0)
    theta = copy(theta0)
    f, g = f_g(theta)
    # parameter-free (the equal-choice null): nothing to optimise, and
    # maximum(abs.(g)) of an empty gradient threw (#495)
    n == 0 && return theta, f, true
    H = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]
    for _ in 1:maxiter
        maximum(abs.(g)) < gtol && return theta, f, true
        _precision_stop(f, g, H; gtol = gtol) && return theta, f, true
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
        ok || return theta, f, _precision_stop(f, g, H; gtol = gtol)
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

"""Refuse mean-design columns that within-choice-set contrasts do not
identify: a covariate common to every alternative, or one duplicating a
generated intercept, lies on an exact likelihood ridge, and BFGS
reported an arbitrary point on it as converged (#581). Left to right, a
column is aliased when its contrast residual on the earlier kept
columns is below `tol` of its own contrast norm (Gram scale)."""
function _check_contrast_rank(m::MNProbit; tol = 1e-10)
    m.p == 0 && return nothing
    m.J >= 2 || throw(ArgumentError(
        "with fewer than two alternatives no mean-design column is " *
        "identified from within-choice-set contrasts (#581)"))
    Dc = reshape(m.X[:, 2:m.J, :] .- m.X[:, 1:1, :], :, m.p)
    G = Dc' * Dc
    kept = Int[]
    aliased = Int[]
    for k in 1:m.p
        gkk = G[k, k]
        resid = gkk
        if gkk > 0 && !isempty(kept)
            g = G[kept, k]
            resid = gkk - LinearAlgebra.dot(
                g, LinearAlgebra.pinv(G[kept, kept]) * g)
        end
        if gkk > 0 && resid > tol * gkk
            push!(kept, k)
        else
            push!(aliased, k)
        end
    end
    isempty(aliased) && return nothing
    nint = m.intercepts ? m.J - 1 : 0
    names = [k <= nint ? "intercept[" * string(k + 1) * "]" :
             "X[:, :, " * string(k - nint) * "]" for k in aliased]
    throw(ArgumentError("mean-design column(s) " * join(names, ", ") *
        " are not identified from within-choice-set contrasts: each is " *
        "constant across the alternatives of every observation or a " *
        "linear combination of earlier columns (the generated intercepts " *
        "come first); drop them (#581)"))
end

"""max c'x s.t. A x <= b, x >= 0 (b >= 0) by the condensed-tableau
simplex with Bland's rule, which terminates on the degenerate b = 0 rows
the separation problem is made of. Returns (x, objective, ok)."""
function _lp_max(c::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64};
                 tol = 1e-12)
    m, n = size(A)
    T = zeros(m + 1, n + 1)
    T[1:m, 1:n] .= A
    T[1:m, n + 1] .= b
    T[m + 1, 1:n] .= -c
    basis = collect((n + 1):(n + m))
    nonbasis = collect(1:n)
    for _ in 1:(50 * (m + n))
        e = 0
        for j in 1:n
            if T[m + 1, j] < -tol && (e == 0 || nonbasis[j] < nonbasis[e])
                e = j
            end
        end
        if e == 0
            x = zeros(n + m)
            for i in 1:m
                x[basis[i]] = T[i, n + 1]
            end
            return x[1:n], T[m + 1, n + 1], true
        end
        r = 0
        best = Inf
        for i in 1:m
            T[i, e] > tol || continue
            ratio = T[i, n + 1] / T[i, e]
            if ratio < best - 1e-15 ||
               (abs(ratio - best) <= 1e-15 && basis[i] < basis[r])
                best = ratio
                r = i
            end
        end
        r == 0 && return zeros(n), Inf, false            # unbounded
        p = T[r, e]
        for i in 1:(m + 1)
            i == r && continue
            f = T[i, e] / p
            f == 0 && continue
            for j in 1:(n + 1)
                j == e && continue
                T[i, j] -= f * T[r, j]
            end
            T[i, e] = -f
        end
        for j in 1:(n + 1)
            j == e && continue
            T[r, j] /= p
        end
        T[r, e] = 1 / p
        basis[r], nonbasis[e] = nonbasis[e], basis[r]
    end
    return zeros(n), 0.0, false
end

"""A recession direction of the mean design, or `nothing`: if some d
gives every observed contrast (X[t, chosen] - X[t, j]) d >= 0, with at
least one strict, every choice probability is nondecreasing along
beta + s d and one rises for ever, so there is no finite MLE (complete
or quasi-complete separation). BFGS stopped where the score fell under
gtol and called it converged (#642). Port of
winning/mnprobit.py::_separation_direction: the LP
max 1'(A d) s.t. A d >= 0, |d|_inf <= 1 on unit-scaled columns."""
function _separation_direction(X::AbstractArray{<:Real,3},
                               choice::AbstractVector{Int})
    T, J, p = size(X)
    (p == 0 || J < 2) && return nothing
    # filled in place: permutedims(hcat(rows...)) splatted one argument
    # per contrast row, and compiling hcat for the Fishing data's 3546
    # arguments took 4.7 s of every fresh-session fit
    A = zeros(T * (J - 1), p)
    row = 0
    for t in 1:T, j in 1:J
        j == choice[t] && continue
        row += 1
        for c in 1:p
            A[row, c] = X[t, choice[t], c] - X[t, j, c]
        end
    end
    cs = [max(maximum(abs, A[:, c]), 0.0) for c in 1:p]
    cs = [x > 0 ? x : 1.0 for x in cs]
    As = A ./ cs'
    nr = size(As, 1)
    g = vec(sum(As, dims = 1))
    Ilp = Matrix{Float64}(LinearAlgebra.I, p, p)
    Alp = vcat(hcat(-As, As), hcat(Ilp, zeros(p, p)), hcat(zeros(p, p), Ilp))
    blp = vcat(zeros(nr), ones(2p))
    x, obj, ok = _lp_max(vcat(g, -g), Alp, blp)
    (ok && obj > 1e-9) || return nothing
    d = x[1:p] .- x[(p + 1):(2p)]
    marg = As * d
    (minimum(marg) >= -1e-9 && maximum(marg) > 1e-7) || return nothing
    d = d ./ cs
    return d ./ LinearAlgebra.norm(d)
end

function fit!(m::MNProbit; method = :exact, maxiter = 400,
              r_draws = 1000, seed = 9)
    maxiter = _as_count(maxiter, "maxiter")
    method == :ghk && _check_r_draws(r_draws)    # before any mutation
    _check_contrast_rank(m)
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
    # With generated intercepts an alternative nobody chose has no finite
    # MLE: its intercept runs to -Inf and BFGS stopped on the score
    # tolerance at a tolerance-dependent point, reported as converged
    # (#496).
    if m.intercepts
        unchosen = [j for j in 1:m.J if !any(==(j), m.choice)]
        if !isempty(unchosen)
            m.converged = false
            @warn "alternative(s) $(unchosen) are never chosen, so their " *
                  "intercepts have no finite MLE (complete separation); " *
                  "the fit is flagged converged = false. Drop those " *
                  "alternatives or fit without intercepts."
        end
    else
        unchosen = Int[]
    end
    sep = _separation_direction(m.X, m.choice)
    m.separation = sep === nothing ? Float64[] : sep
    if sep !== nothing && isempty(unchosen)
        m.converged = false
        @warn "the mean design is separated: moving the coefficients " *
              "along $(round.(sep, digits = 6)) never lowers any observed " *
              "choice probability and raises at least one, so the " *
              "likelihood has no finite maximizer; the fit is flagged " *
              "converged = false (see m.separation, #642)."
    end
    # the fit maximized raw per-choice masses; prediction normalizes
    # them: report how far apart the two maps are (#376)
    m.mass_defect = maximum(abs, mass_defect(m))
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

"""Refuse inference in a singular loading chart. The free loadings are
lower triangular with a zero reference row, which pins the factor
rotation only while each pivot V[c + 1, c], c < r, is nonzero. When
alternative c + 1 loads like the reference on the first c factors, a
continuous rotation of the factors keeps every imposed zero and every
probability: the exact likelihood is flat along it, and the finite
quadrature's small artificial curvature turned that ridge into large
but finite standard errors whose very existence depended on the
alternative ORDER (condition number 4384 against 19 for a relabelled,
identical covariance, #645)."""
function _check_chart(m::MNProbit; tol = 1e-6)
    m.r >= 2 || return nothing
    scale = max(maximum(abs, m.V; init = 0.0), 1.0)
    for c in 1:(m.r - 1)
        if abs(m.V[c + 1, c]) <= tol * scale
            throw(ArgumentError(
                "vcov: the loading chart is singular at its pivot V[" *
                string(c + 1) * ", " * string(c) * "] = " *
                string(m.V[c + 1, c]) * ": alternative " * string(c + 1) *
                " loads like the reference on the first " * string(c) *
                " factor(s), so a factor rotation leaves every probability " *
                "unchanged and the raw loadings have no standard errors. " *
                "Reorder the alternatives so one with a distinct loading " *
                "takes row " * string(c + 1) * " and refit, or pass " *
                "allow_singular_chart = true to see the quadrature's " *
                "pseudo-information anyway (#645)"))
        end
    end
    return nothing
end

"""Parameter covariance at the fit: method = :hessian (observed
information, default), :opg (outer product of per-observation
scores), or :sandwich (H^-1 B H^-1, robust).

`:opg` never touches the Hessian. It used to compute and invert the
observed information first and only then branch, so a singular or
failing Hessian blocked the OPG fallback that exists for exactly that
case, and paid 2 * nparams score evaluations for nothing (#434).

Asymptotic covariance is a statement about a MAXIMUM. A non-converged
fit is refused unless `allow_unconverged = true`, and the Hessian and
sandwich forms require the observed information `-H` to be positive
definite: at a saddle its inverse has negative variances, which
`stderror` used to clip to a standard error of exactly zero (#493)."""
function vcov(m::MNProbit; method = :hessian, allow_unconverged = false,
              allow_singular_chart = false)
    method in (:hessian, :opg, :sandwich) ||
        error("method must be :hessian, :opg or :sandwich")
    _require_exact(m, "vcov")
    allow_singular_chart || _check_chart(m)
    (m.converged || allow_unconverged) || throw(ArgumentError(
        "vcov: the fit did not converge, so theta is not a maximum and " *
        "its curvature is not a covariance. Refit (e.g. larger maxiter) " *
        "or pass allow_unconverged = true to inspect it anyway"))
    if method == :opg
        G = score_matrix(m)
        return inv(G' * G)
    end
    info = Symmetric(-loglik_hessian(m))
    ch = cholesky(info; check = false)
    issuccess(ch) || throw(ArgumentError(
        "vcov: the observed information is not positive definite (smallest " *
        "eigenvalue " * string(round(eigmin(info), sigdigits = 4)) * "), so " *
        "theta is not a strict local maximum and -H^-1 is not a " *
        "covariance; method = :opg does not use the Hessian"))
    Hinv = inv(ch)
    method == :hessian && return (Hinv .+ Hinv') ./ 2
    G = score_matrix(m)
    return Hinv * (G' * G) * Hinv
end

"""Standard errors: square roots of the `vcov` diagonal. A negative
variance is refused, not clipped to a zero standard error (#493); only
round-off, below `1e-12` of the largest diagonal entry, is read as 0."""
function stderror(m::MNProbit; method = :hessian, allow_unconverged = false,
                  allow_singular_chart = false)
    d = LinearAlgebra.diag(vcov(m; method = method,
                                allow_unconverged = allow_unconverged,
                                allow_singular_chart = allow_singular_chart))
    tol = 1e-12 * maximum(abs, d; init = 0.0)
    for (i, x) in enumerate(d)
        (isfinite(x) && x >= -tol) || throw(ArgumentError(
            "stderror: variance " * string(i) * " is " * string(x) *
            "; a negative variance is not a standard error"))
    end
    return sqrt.(max.(d, 0.0))
end

function Base.show(io::IO, ::MIME"text/plain", m::MNProbit)
    println(io, "MNProbit  J=", m.J, " T=", m.T, " r=", m.r,
            "  method=", m.method,
            "  logLik=", round(m.loglik, digits = 3),
            m.converged ? "" : "  (NOT converged)")
    isnan(m.loglik) && return
    # standard errors exist only for a converged exact fit (#214, #493)
    exact = m.method == :exact && m.converged
    se = !exact ? fill(NaN, length(m.theta)) : try
        stderror(m)
    catch
        fill(NaN, length(m.theta))
    end
    names = vcat(["beta[$i]" for i in 1:m.p],
                 ["v[$row,$col]" for (row, col) in m.pos])
    println(io, rpad("param", 12), rpad("estimate", 12),
            exact ? "se" :
            m.method == :exact ? "se (not available: the fit did not converge)" :
            "se (not available for a :" * string(m.method) * " fit)")
    for i in eachindex(m.theta)
        println(io, rpad(names[i], 12),
                rpad(string(round(m.theta[i], digits = 4)), 12),
                exact ? round(se[i], digits = 4) : "-")
    end
end

"""Choice probabilities under the fitted parameters, by the same
factor-conditional product integrals the likelihood uses. The raw
per-alternative masses do not sum to one exactly; `normalize = true`
(default) divides by the row total, `normalize = false` returns the raw
masses `fit!` maximized, and `mass_defect` reports the gap (#376).

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
function predict_proba(m::MNProbit; X = nothing, normalize = true)
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
    # the likelihood's own per-observation map, so prediction prices a
    # model exactly as fitting does (the same tilt, blend and closed
    # form); this copy had its own untilted rule and a 1e-300 floor
    P = zeros(T2, m.J)
    Dv = ones(m.J)
    for k in 1:m.J
        lp, _, _ = _choice_logprob_terms(mu, m.V, fill(k, T2), Dv)
        P[:, k] .= exp.(lp)
    end
    normalize || return P
    return P ./ sum(P, dims = 2)
end

"""`1 - sum_k raw P_k` per observation: the gap between the raw
per-choice masses `fit!` maximizes and the normalized `predict_proba`
(#376). `fit!` stores the largest absolute value on the training rows in
`m.mass_defect`."""
mass_defect(m::MNProbit; X = nothing) =
    1 .- vec(sum(predict_proba(m; X = X, normalize = false), dims = 2))

end # module
