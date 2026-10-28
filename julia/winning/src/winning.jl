# The winning Julia port: factor-correlated races (min-wins) and the
# top-k membership module, transliterated from the R reference
# (r/winning/R/races.R, topk.R); the python numpy implementation is
# the spec and parity/check.jl pins this port to its vectors.
#
# Scope so far: independent and factor races (forward + slopes +
# inversion), the full top-k surface (pricing, rank marginals, both
# Jacobians, the four calibrations). The covariance grammar
# (blocks/nested/tree), polish and the classic lattice remain on the
# roadmap.
module winning

using LinearAlgebra: SymTridiagonal, eigen

export race_probabilities, abilities_from_race, hermite_nodes,
    top_k_probabilities, bottom_k_probabilities, rank_probabilities,
    top_k_jacobians, abilities_from_topk, loc_scale_from_topk_pair,
    loc_scale_from_win_and_second, abilities_from_rank_marginal

const TINY = 1e-300
const EULER = 0.5772156649015329

# ---- normal cdf: series + scaled complementary erf (port of the JS
# core, itself pinned to scipy) --------------------------------------

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
    x = z / sqrt(2)
    x >= 2.5 && return 1 - 0.5 * _erfcx(x) * exp(-x * x)
    x <= -2.5 && return 0.5 * _erfcx(-x) * exp(-x * x)
    return 0.5 * (1 + _erf_series(x))
end

npdf(z) = exp(-0.5 * z * z) / sqrt(2 * pi)

function qnorm(p::Float64)
    # Acklam rational approximation, adequate for node placement
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
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    end
    p > 1 - pl && return -qnorm(1 - p)
    q = p - 0.5
    r2 = q * q
    return (((((a[1] * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * r2 + a[6]) * q /
           (((((b[1] * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + b[5]) * r2 + 1)
end

# ---- base families: z -> (S, f, fp), elementwise over arrays -------

function base_normal(z)
    # the upper tail directly: 1 - ndtr(z) cancels to exactly 0 past about
    # 8.3 sd, and a rank-4 last place of [-12, -4, 4, 12] came back
    # 5.5e-72 for R's and python's 4.2e-68 (#96)
    S = max.(ndtr.(.-z), TINY)
    f = npdf.(z)
    return (S = S, f = f, fp = -z .* f)
end

"""log of the normal survival where `base_normal` floored it at TINY
(z past 37): log(erfcx(z/sqrt 2)/2) - z^2/2, as R's pnorm(log.p = TRUE)
and python's log_ndtr(-z) (#502). Used only at floored points, so
nothing above 1e-300 moves."""
_log_sf_normal(z::Float64) = z > 0 ? log(0.5 * _erfcx(z / sqrt(2))) - 0.5 * z * z :
                                     log(ndtr(-z))

"""log S for the race integrator: log.(S), except where the normal base
floored S at TINY, which take its exact log survival (R's .log_survival,
python's _log_survival). The floor plateaued a rare winner at ~1e-300."""
function _log_survival(fn, z, S)
    logS = log.(S)
    fn === base_normal || return logS
    for i in eachindex(S)
        S[i] <= TINY && (logS[i] = max(_log_sf_normal(z[i]), -1e300))
    end
    return logS
end

function base_gumbel(z)
    c = pi / sqrt(6)
    u = min.(z .* c .- EULER, 30)
    eu = exp.(u)
    S = max.(exp.(-eu), TINY)
    f = c .* eu .* S
    return (S = S, f = f, fp = c * c .* eu .* S .* (1 .- eu))
end

function base_logistic(z)
    c = pi / sqrt(3)
    u = clamp.(c .* z, -700, 700)
    S = 1 ./ (1 .+ exp.(u))
    f = c .* S .* (1 .- S)
    return (S = max.(S, TINY), f = f, fp = -c .* f .* (1 .- 2 .* S))
end

function base_laplace(z)
    b = 1 / sqrt(2)
    f = exp.(-abs.(z) ./ b) ./ (2b)
    S = ifelse.(z .< 0, 1 .- 0.5 .* exp.(z ./ b), 0.5 .* exp.(-z ./ b))
    return (S = max.(S, TINY), f = f, fp = -sign.(z) .* f ./ b)
end

const BASES = Dict("normal" => base_normal, "gumbel" => base_gumbel,
                   "logistic" => base_logistic, "laplace" => base_laplace)
const SPANS = Dict("normal" => (8.0, 8.0), "gumbel" => (22.0, 8.0),
                   "logistic" => (16.0, 16.0), "laplace" => (18.0, 18.0))

_base_fn(base) = base isa Function ? base : BASES[base]

# ---- Gauss-Hermite (probabilists'), Golub-Welsch -------------------

function hermite1(order::Int)
    T = SymTridiagonal(zeros(order), sqrt.(1.0:(order - 1)))
    E = eigen(T)
    w = E.vectors[1, :] .^ 2
    return (nodes = E.values, weights = w ./ sum(w))
end

"""
    hermite_nodes(k, order = 15, prune = 1e-7)

Pruned Gauss-Hermite product rule over N(0, I_k), first coordinate
slowest, pruned THEN renormalized so the weights sum to one, matching the
Python reference and the R port (#802: this used to say the reference did
not renormalize, and `hermite_nodes(2, 5, 0.5).W` summed to 0.284).

`k = 0` is the EMPTY PRODUCT: one node of weight 1 with no columns, so a
zero-rank loading matrix integrates over the zero-dimensional factor
space and prices the independent race through this same path. `k < 0`
and `order < 1` raised from inside `ntuple`/`Iterators.product` with a
message about tuples rather than about the rule being asked for (#68).
"""
function hermite_nodes(k::Int, order::Int = 15, prune::Float64 = 1e-7)
    k < 0 && throw(ArgumentError("hermite_nodes needs k >= 0; got $k"))
    order < 1 && throw(ArgumentError(
        "hermite_nodes needs order >= 1; got $order"))
    k == 0 && return (F = Matrix{Float64}(undef, 1, 0), W = [1.0])
    h = hermite1(order)
    k == 1 && return (F = reshape(h.nodes, :, 1), W = copy(h.weights))
    idx = [(i, j) for i in 1:order for j in 1:order]  # first slowest, k = 2
    if k == 2
        F = [h.nodes[t[c]] for t in idx, c in 1:2]
        W = [h.weights[t[1]] * h.weights[t[2]] for t in idx]
    else
        combos = Iterators.product(ntuple(_ -> 1:order, k)...)
        rows = [reverse(collect(t)) for t in vec(collect(combos))]
        F = [h.nodes[r[c]] for r in rows, c in 1:k]
        W = [prod(h.weights[r]) for r in rows]
    end
    keep = W .> prune * maximum(W)
    Wk = W[keep]
    return (F = F[keep, :], W = Wk ./ sum(Wk))
end

"""The first `n` primes, generated. `halton_normal_nodes` used a fixed
12-prime table, and the automatic rule sends any rank with 15^r > 1e5 to
Halton, so a full-contrast-rank race at r = 13 raised a BoundsError at
`primes[13]` (#402)."""
function _first_primes(n::Int)
    ps = Int[]
    c = 2
    while length(ps) < n
        isp = true
        for p in ps
            p * p > c && break
            c % p == 0 && (isp = false; break)
        end
        isp && push!(ps, c)
        c += 1
    end
    return ps
end

function halton_normal_nodes(r::Int, n::Int)
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
            F[row, c] = qnorm(clamp(h, 1e-12, 1 - 1e-12))
        end
    end
    return (F = F, W = fill(1.0 / n, n))
end

# ---- race setup (adaptive factor quadrature, matching R) -----------

# A scalar D is the same variance for everyone, as as_idio states;
# a zero variance is a contestant with no idiosyncratic noise, which
# this lattice cannot represent -- it returned NaN where python, R
# and the browser all refuse. Found by the cross-port divergence
# scan. Shared by the winner race and every top-k/rank entry point,
# which used to collect(D) themselves and failed on a scalar (#485).
function _as_idio(D, n::Integer; name = "D")
    D === nothing && return ones(n)
    Dv = D isa Number ? [Float64(D)] : Float64.(vec(collect(D)))
    length(Dv) == 1 && (Dv = fill(Dv[1], n))
    length(Dv) == n || throw(ArgumentError(
        name * " must be a scalar or one idiosyncratic variance per " *
        "contestant; got " * string(length(Dv)) * " for " * string(n)))
    all(isfinite, Dv) || throw(ArgumentError(name * " has a non-finite entry"))
    all(>(0.0), Dv) || throw(ArgumentError(
        name * " must be strictly positive here: a zero variance is a " *
        "contestant the lattice cannot represent"))
    return Dv
end

"""mu moved next to the origin when a common offset dwarfs the field, as
python's races._translated_home. The origin is a gauge, but the lattice is
laid in the caller's units: at an offset of 2^52 its points collapsed to
the float spacing there, so race shares came back NaN and top-k/rank
refused a total membership of 0 (#477). Shifting by a middle entry (an
element, so no sum can overflow) is exact for such offsets; below a
million field scales nothing moves, so ordinary inputs stay bit-identical.
A comparison with NaN is false, so non-finite mu passes through untouched."""
function _translated_home(mu, D)
    isempty(mu) && return mu
    scale = (maximum(mu) - minimum(mu)) + sqrt(maximum(D))
    # |off| <= max|mu|: an ordinary field skips the sort
    maximum(abs, mu) > 1e6 * scale || return mu
    off = sort(mu)[length(mu) ÷ 2 + 1]
    return abs(off) > 1e6 * scale ? mu .- off : mu
end

# The rank-one Gauss-Hermite/midpoint handover was a hard switch at
# ceil(8 sharp) > 80, where the two rules disagree: a 1e-7 change in one
# runner's sd moved a top-2 membership by 31bp (#783). Across (lo, hi]
# the rule is the weighted union (1 - w) GH(8 lo) + w midpoint(8 hi + 1),
# w a C1 smoothstep in the sharpness, matching the python reference
# (which says why the band sits on the Gauss-Hermite side); each order is
# the one its side uses at the band's edge, so the band meets the
# unchanged rule outside it continuously at both ends.
const R1_BLEND_LO = 9.0
const R1_BLEND_HI = 10.0
function _rank1_blend_rule(sharp)
    t = (sharp - R1_BLEND_LO) / (R1_BLEND_HI - R1_BLEND_LO)
    w = t * t * (3.0 - 2.0 * t)
    g = hermite_nodes(1, Int(8 * R1_BLEND_LO))
    Q = Int(8 * R1_BLEND_HI) + 1
    Fm = reshape(qnorm.(((1:Q) .- 0.5) ./ Q), :, 1)
    return vcat(g.F, Fm), vcat((1.0 - w) .* g.W, fill(w / Q, Q))
end

# Below the band the rank-one rule was Gauss-Hermite of order
# ceil(8 sharp), whose prices alternate with the parity of the order and so
# jumped at every 1/8 of sharpness (#850). Matching the python reference
# (_rank1_hermite_rule, which has the measurements), with x = 8 sharp and S
# the smoothstep above: on (15, 24] the adjacent orders interpolated,
# (1 - w) GH(m) + w GH(m+1); on (24, 25] GH(24) takes on GH(23) up to half
# weight; past 25 the parity-balanced pair GH(m-1)/2 + GH(m)/2 at order
# y = 5 sharp in [24, 34], sliding its lower half to GH(m+1) as y crosses
# (m, m+1); on (71, 72] handed back to GH(72), the band's lower edge.
const R1_PAIR_FROM = 24
const R1_PAIR_PER_SHARP = 5.0
const R1_PAIR_CAP = 34
function _rank1_hermite_rule(sharp)
    S(t) = t * t * (3.0 - 2.0 * t)
    x = 8.0 * sharp
    lo = R1_PAIR_FROM
    if x <= lo
        m = floor(x); w = S(x - m)
        parts = [(Int(m), 1.0 - w), (Int(m) + 1, w)]
    elseif x <= lo + 1
        w = S(x - lo)
        parts = [(lo - 1, 0.5 * w), (lo, 1.0 - 0.5 * w)]
    else
        y = min(max(R1_PAIR_PER_SHARP * sharp, Float64(lo)), Float64(R1_PAIR_CAP))
        m = floor(y); w = S(y - m)
        parts = [(Int(m) - 1, 0.5 * (1.0 - w)), (Int(m), 0.5), (Int(m) + 1, 0.5 * w)]
        top = Int(8 * R1_BLEND_LO)
        if x > top - 1
            v = S(x - (top - 1))
            parts = vcat([(q, c * (1.0 - v)) for (q, c) in parts], [(top, v)])
        end
    end
    parts = [(q, c) for (q, c) in parts if c > 0.0]
    g = [hermite_nodes(1, q) for (q, _) in parts]
    return vcat([h.F for h in g]...), vcat([c .* h.W for (h, (_, c)) in zip(g, parts)]...)
end

function _race_setup(mu, V, D, F, W, base)
    mu = Float64.(collect(mu))
    n = length(mu)
    D = _as_idio(D, n)
    mu = _translated_home(mu, D)                    # (#477)
    if V === nothing
        Vm = zeros(n, 1)
        Fm = zeros(1, 1)
        Wv = [1.0]
    else
        # a SCALAR V is the same loading for everyone -- the spelling
        # as_loadings documents, which python and the browser accept
        # and julia refused (cross-port divergence scan)
        Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) :
             (V isa Number ? fill(Float64(V), n, 1) :
              reshape(Float64.(collect(V)), :, 1))
        # (rank, n) is the SAME race as (n, rank), which as_loadings
        # documents. The ambiguity is real only at rank == n, where the
        # contract wins: n rows is n contestants.
        if size(Vm, 1) != n && size(Vm, 2) == n
            Vm = permutedims(Vm)
        end
        size(Vm, 1) == n || throw(ArgumentError(
            "V must have one row per contestant; got " *
            string(size(Vm, 1)) * " for " * string(n)))
        Vm = _gauge_center(Vm)                 # common column is gauge (#471)
        if F === nothing || W === nothing
            sharp = sqrt(2) * maximum(sqrt.(vec(sum(Vm .^ 2, dims = 2))) ./
                                      sqrt.(max.(D, TINY)))
            r = size(Vm, 2)
            # per-rank (Gauss-Hermite order cap, sharpness past which even
            # that order loses to the low-discrepancy family): GH_RULE in
            # the python reference, identical in R and the browser. Julia
            # kept the older single 3.0 cutover, rank-3 cap 15 and the
            # rank-one handover at Q > 201, where GH-169 was 42bp off on
            # the 8-runner stress field at sharpness 21 (#528, #153).
            cap = r == 1 ? 201 : r == 2 ? 41 : r == 3 ? 31 : 15
            sharp_max = r == 2 ? 3.75 : r == 3 ? 4.75 : 3.0
            if r >= 2 && sharp > sharp_max
                hw = halton_normal_nodes(r, 2^13)
                Fm, Wv = hw.F, hw.W
            elseif r == 1 && R1_BLEND_LO < sharp <= R1_BLEND_HI
                Fm, Wv = _rank1_blend_rule(sharp)              # (#783)
            elseif r == 1 && ceil(8 * sharp) > 80
                Q = Int(min(ceil(8 * sharp), 4001))
                Fm = reshape(qnorm.(((1:Q) .- 0.5) ./ Q), :, 1)
                Wv = fill(1.0 / Q, Q)
            elseif r == 1 && 8 * sharp > 15
                Fm, Wv = _rank1_hermite_rule(sharp)            # (#850)
            elseif Float64(cap)^r > 1e5
                hw = halton_normal_nodes(r, 2^13)
                Fm, Wv = hw.F, hw.W
            else
                Q = Int(min(max(ceil(8 * sharp), 15), cap))
                hw = hermite_nodes(size(Vm, 2), Q)
                Fm, Wv = hw.F, hw.W
            end
        else
            # The caller's nodes go through the same door as V and D:
            # every node carries exactly the loadings' rank, and there
            # is one weight per node. An F with the wrong number of
            # ROWS was accepted and priced a different quadrature
            # outright -- [0.646, 0.123, 0.231, 0.0005] where the right
            # answer is [0.382, 0.301, 0.137, 0.181] -- and an extra
            # weight was silently ignored. The two that did fail failed
            # with a DimensionMismatch or a BoundsError from somewhere
            # inside, which names nothing the caller passed (#290, in
            # the browser; this is julia's copy).
            Fm = F isa AbstractMatrix ? Float64.(Matrix(F)) :
                 reshape(Float64.(collect(F)), :, 1)
            rank = size(Vm, 2)
            size(Fm, 2) == rank || throw(ArgumentError(
                "F must have one column per loading; got " *
                string(size(Fm, 2)) * " for rank " * string(rank) *
                " -- a short F prices a lower-rank model"))
            size(Fm, 1) >= 1 || throw(ArgumentError(
                "F is empty; there are no quadrature nodes"))
            all(isfinite, Fm) || throw(ArgumentError("F has a non-finite node"))
            Wv = Float64.(collect(W))
            length(Wv) == size(Fm, 1) || throw(ArgumentError(
                "W must have one weight per factor node; got " *
                string(length(Wv)) * " for " * string(size(Fm, 1)) * " nodes"))
            all(isfinite, Wv) || throw(ArgumentError("W has a non-finite weight"))
            all(>=(0.0), Wv) || throw(ArgumentError(
                "W has a negative weight; a factor law has no negative mass"))
            # W and c*W are the same law. Normalise through the LARGEST
            # weight: finite weights such as [1e308, 1e308] overflow only
            # in the raw total, and W ./ Inf made every weight zero and
            # every probability NaN (#415).
            wmax = maximum(Wv)
            wmax > 0 || throw(ArgumentError(
                "W must have a positive total; got " * string(sum(Wv))))
            Wv = Wv ./ wmax
            Wv = Wv ./ sum(Wv)
        end
    end
    fn = _base_fn(base)
    span = base isa Function ? (12.0, 12.0) : get(SPANS, base, (12.0, 12.0))
    return (mu = mu, V = Vm, D = D, F = Fm, W = Wv, fn = fn,
            left = span[1], right = span[2])
end

function _bulk_window(M_all, sd, points, delta)
    mu_lo = vec(minimum(M_all, dims = 1))
    mu_hi = vec(maximum(M_all, dims = 1))
    G(x) = 1 - exp(sum(log.(max.(1 .- ndtr.((x .- mu_lo) ./ sd), TINY))))
    H(x) = 1 - exp(sum(log.(max.(1 .- ndtr.((x .- mu_hi) ./ sd), TINY))))
    lo0 = minimum(mu_lo) - 9 * maximum(sd)
    hi0 = maximum(mu_hi) + 9 * maximum(sd)
    a, b = lo0, hi0
    for _ in 1:80
        m = 0.5 * (a + b)
        G(m) < delta ? (a = m) : (b = m)
    end
    xlo = a
    a, b = xlo, hi0
    for _ in 1:80
        m = 0.5 * (a + b)
        H(m) < 1 - delta ? (a = m) : (b = m)
    end
    pad = 2 * maximum(sd)
    return range(xlo - pad, b + pad, length = points)
end

"""Win probabilities of the general race, all N in one field pass per
factor node; optionally the raw own-location slopes (the inversion
preconditioner), both normalized by the total."""
function race_probabilities(mu; V = nothing, D = nothing, F = nothing,
                            W = nothing, base = "normal", points = 257,
                            return_slopes = false, window = "bulk",
                            delta = 1e-12)
    _check_points(points)
    # the bulk window's tail mass: a number in (0, 0.5); delta = -1 hung
    # python's and the browser's relaxation loop and 1 or 2 priced a
    # different race (#445)
    (delta isa Real && !(delta isa Bool) && isfinite(delta) &&
     0 < delta < 0.5) ||
        throw(ArgumentError("delta must be a number in (0, 0.5); got $delta"))
    st = _race_setup(mu, V, D, F, W, base)
    n = length(st.mu)
    if n == 2 && base == "normal" && F === nothing
        # A two-runner normal race is ONE Gaussian contrast (python's and
        # R's pair branch): p1 = Phi((mu2 - mu1) / sd_d), sd_d^2 =
        # |V1 - V2|^2 + D1 + D2 under the default rule, which stands for
        # the standard normal law. The lattice priced a 20-sd pair tail at
        # 0.0105x its exact 1.04e-45 (#86). A caller's nodes need not be
        # Gaussian, so they keep the lattice.
        dV = st.V[1, :] .- st.V[2, :]
        sh = sum(abs2, dV)
        big = max(abs(sh), st.D[1], st.D[2])
        sd_d = big > 0 ? sqrt(big) * sqrt(max(sh / big + (minimum(st.D) / big +
                                                        maximum(st.D) / big), 0.0)) : 0.0
        if !isfinite(sh)
            # a finite contrast past 1e154 squares to Inf: the norm and u
            # in units of the largest term, never squared at full scale.
            # The norm itself may pass the float range (rank two at
            # 7e307 gave a = Inf, NaN and then [1, 0] for Phi(1/sqrt 2),
            # #86); python's pair branch
            m = max(maximum(abs, dV), sqrt(maximum(st.D)))
            rel = sqrt(sum(abs2, dV ./ m) + st.D[1] / m / m + st.D[2] / m / m)
            sd_d = m * rel
            u = (st.mu[2] / m - st.mu[1] / m) / rel
        else
            sd_d > 0 || (sd_d = 1e-150)
            u = (st.mu[2] - st.mu[1]) / sd_d
        end
        # each tail directly: 1 - ndtr(u) rounds a 9-sd loser to zero
        p = [ndtr(u), ndtr(-u)]
        if return_slopes
            dens = npdf(u) / sd_d
            return (p = p, slopes = [-dens, -dens])
        end
        return p
    end
    sd = sqrt.(st.D)
    Q = size(st.F, 1)
    M_all = repeat(st.mu', Q, 1) .+ st.F * st.V'
    x = if window == "bulk"
        collect(_bulk_window(M_all, sd, points, delta))
    else
        collect(range(minimum(M_all) - st.left * maximum(sd),
                      maximum(M_all) + st.right * maximum(sd),
                      length = points))
    end
    dx = x[2] - x[1]
    smin = minimum(sd)
    sharp_here = maximum(sqrt.(vec(sum(st.V .^ 2, dims = 2)))) / max(smin, TINY)
    if sharp_here > 25 && dx > 0.5 * smin
        need = Int(ceil((x[end] - x[1]) / (0.5 * smin))) + 1
        pts2 = min(need, 8193)
        if pts2 > points
            x = collect(range(x[1], x[end], length = pts2))
            dx = x[2] - x[1]
        end
        need > 8193 && @warn "conditional races sharper than the lattice can resolve"
    end
    L = length(x)
    p = zeros(n)
    slope = zeros(n)
    for q in 1:Q
        z = (x' .- M_all[q, :]) ./ sd            # (n, L)
        b = st.fn(z)
        f = b.f ./ sd
        logS = _log_survival(st.fn, z, b.S)
        Lfield = vec(sum(logS, dims = 1))        # per lattice point
        rest = exp.(clamp.(Lfield' .- logS, -745, 0))
        p .+= st.W[q] .* vec(sum(f .* rest, dims = 2)) .* dx
        slope .+= st.W[q] .* vec(sum(-b.fp ./ sd .^ 2 .* rest, dims = 2)) .* dx
    end
    total = sum(p)
    return_slopes && return (p = p ./ total, slopes = slope ./ total)
    return p ./ total
end

# A lattice size is a whole number >= 2: x[2] - x[1] raised BoundsError
# at 1 and 0 (#444), the contract python's shapes.as_points states.
function _check_points(points)
    (points isa Real && isfinite(points) && points == floor(points)) ||
        throw(ArgumentError("points must be a whole number of lattice points; got $points"))
    points >= 2 ||
        throw(ArgumentError("points must be at least 2: a lattice of $points point(s) has no spacing"))
    return Int(points)
end

# The gauge, scale-safe (python core._gauge_center, #471): an exactly
# common column centres to exactly zero, and a column whose mean or
# differences overflow ([1e308] * 9, or [1, 1, -1] * 1e308) is centred in
# units of its largest entry. Ordinary columns take the plain mean.
function _gauge_center(V::AbstractMatrix)
    n = size(V, 1)
    n == 0 && return Float64.(V)
    Vc = V .- sum(V, dims = 1) ./ n
    for j in 1:size(V, 2)
        col = view(V, :, j)
        if all(==(col[1]), col)
            Vc[:, j] .= 0.0
        elseif !all(isfinite, view(Vc, :, j))
            s = maximum(abs, col)
            u = col ./ s
            Vc[:, j] .= (u .- sum(u) / n) .* s
        end
    end
    return Vc
end

"""Invert the general race: mean-zero mu with matching probabilities,
by the damped own-slope iteration with residual-proportional caps,
in the field's own units (#100)."""
function abilities_from_race(p; V = nothing, D = nothing, F = nothing,
                             W = nothing, base = "normal", points = 257,
                             n_iter = 60, tol = 1e-8)
    target = Float64.(collect(p))
    all(isfinite, target) || error("target probabilities must be finite")
    any(target .<= 0) && error("all target probabilities must be positive")
    # a target is a law up to a positive factor: when its SUM overflows
    # (entries all finite) rescale by the max first, as python does since
    # #300; conditional, so ordinary inputs stay bit-identical (#326)
    tot = sum(target)
    if !isfinite(tot)
        target ./= maximum(target)
        tot = sum(target)
    end
    target ./= tot
    logt = log.(target)
    n = length(target)
    # The field's contrast scale, as python and R measure it: the warm
    # start and the caps were unit-variance numbers, so D = [.005, .01,
    # .02] ended at a log residual of 51.6 while the same race at
    # D * 100^2 converged (#100).
    st = _race_setup(zeros(n), V, D, F, W, base)
    Dn = sort(st.D)
    medD = isodd(n) ? Dn[(n + 1) ÷ 2] : 0.5 * (Dn[n ÷ 2] + Dn[n ÷ 2 + 1])
    Vc = _gauge_center(st.V)
    scale = sqrt(medD + sum(Vc .^ 2) / n)
    mu = -(logt .- sum(logt) / n) ./ 2 .* scale
    # damped for a pair AND for a dominant pair at any n (top-two share
    # above 0.8), whose Jacobi step two-cycles -- python's gate
    top2 = n > 2 ? sum(sort(target)[end-1:end]) : 1.0
    alpha = (n == 2 || top2 > 0.8) ? 0.7 : 1.0
    resid_max = Inf
    for _ in 1:n_iter
        ps = race_probabilities(mu; V = V, D = D, F = F, W = W,
                                base = base, points = points,
                                return_slopes = true)
        resid = log.(max.(ps.p, TINY)) .- logt
        resid_max = maximum(abs.(resid))
        resid_max < tol && break
        dlogp = min.(ps.slopes ./ max.(ps.p, TINY), -1e-6 / scale)
        lim = min.(2, 10 .* abs.(resid)) .* scale
        mu .-= clamp.(alpha .* resid ./ dlogp, -lim, lim)
        mu .-= sum(mu) / length(mu)
    end
    # it used to stop at n_iter and return a plausible vector silently
    resid_max < tol ||
        @warn "abilities_from_race did not converge" resid_max n_iter tol
    return mu
end

"""Own-slope-preconditioned coordinate (Jacobi) sweeps on the mean-zero
quotient, mu <- mu - alpha resid / dres, capped and recentred: python's
`races._jacobi_sweeps`, line for line (R has `.jacobi_sweeps`).
`forward(mu)` returns (resid, dres), the residual in the caller's space
and its own-slopes (negative). Damping adapts to the contraction
observed: consecutive steps estimate the dominant Jacobi eigenvalue and
set the Richardson alpha (2/3 for a pair's two-cycle), a sweep that
contracts neither the max nor the rms residual is undone under a
transient penalty, and collinear geometrically decaying steps are summed
(Aitken) as a capped, repriced trial while the residual is still a
thousand tolerances out. Returns (mu, converged, resid_max, iters)."""
function _jacobi_sweeps(mu, forward, scale, alpha, n_iter, tol)
    mu = copy(mu)
    resid_max = Inf
    iters = 0
    prev = nothing
    prev_step = nothing
    alpha_base = Float64(alpha)
    penalty = 1.0
    cached = nothing
    for it in 1:n_iter
        iters = it
        if cached !== nothing
            resid, dres = cached
            cached = nothing
        else
            resid, dres = forward(mu)
        end
        resid_max = maximum(abs.(resid))
        resid_rms = sqrt(sum(abs2, resid) / length(resid))
        resid_max < tol && break
        if prev !== nothing && resid_max >= prev[4] && resid_rms >= prev[5]
            if penalty > 0.1
                penalty = max(0.5 * penalty, 0.1)
                mu, resid, dres, resid_max, resid_rms = prev
                prev_step = nothing
            end
        elseif prev !== nothing && penalty < 1.0
            penalty = min(1.0, penalty / 0.75)
        end
        prev = (mu, resid, dres, resid_max, resid_rms)
        a = alpha_base * penalty
        lim = min.(2.0, 10.0 .* abs.(resid)) .* scale
        step = clamp.(a .* resid ./ dres, -lim, lim)
        step .-= sum(step) / length(step)
        if prev_step !== nothing
            na = sqrt(sum(abs2, prev_step))
            nb = sqrt(sum(abs2, step))
            if na > 0
                dot = sum(step .* prev_step)
                rho = dot / (na * na)
                cosn = nb > 0 ? dot / (na * nb) : 0.0
                ratio = nb / na
                if rho < 0
                    lam = 1.0 - (1.0 - rho) / a
                    alpha_base = clamp(2.0 / (2.0 - lam), 0.1, 1.0)
                elseif cosn > 0.999 && 0.5 < ratio < 0.999 &&
                       resid_max > 1e3 * tol
                    trial = _aitken_trial(mu, step, ratio, scale, forward,
                                          resid_max)
                    if trial !== nothing      # sum the geometric tail
                        mu, cached = trial
                        prev_step = nothing
                        continue
                    end
                end
            end
        end
        prev_step = step
        mu = mu .- step
    end
    if !(resid_max < tol)
        # exhausted after a step no sweep priced: price the returned
        # iterate (one forward, only here) and keep the better (#497)
        last = prev === nothing ? (mu, resid_max) : (prev[1], prev[4])
        fwd = cached
        if fwd === nothing
            # the last step crossed the forward's refusal guard: return the
            # last priced iterate as not converged; with nothing priced it
            # still throws (#823)
            fwd = try
                forward(mu)
            catch e
                # only the forward's refusal (error(...), an ErrorException,
                # as python catches only RuntimeError): a MethodError or
                # BoundsError is a bug and must propagate, not be reported
                # as converged = false
                (prev === nothing || !(e isa ErrorException)) && rethrow()
                return prev[1], false, prev[4], iters
            end
        end
        r_end = maximum(abs.(fwd[1]))
        if prev !== nothing && !(r_end < last[2])
            mu = last[1]
        else
            resid_max = r_end
        end
    end
    return mu, resid_max < tol, resid_max, iters
end

"""The Aitken-extrapolated iterate as a bounded, checked trial (#583),
python's `races._aitken_trial`: the summed tail capped in the max norm at
twice the field scale (never below the ordinary step) and repriced;
`nothing` -- take the ordinary step -- when the forward throws, is
non-finite, or its max residual grows tenfold."""
function _aitken_trial(mu, step, ratio, scale, forward, resid_max)
    jump = step ./ (1.0 - ratio)
    big = maximum(abs.(jump))
    cap = max(2.0 * maximum(scale), maximum(abs.(step)))
    big > cap && (jump = jump .* (cap / big))
    cand = mu .- jump
    fc = try
        forward(cand)
    catch
        return nothing
    end
    (all(isfinite, fc[1]) && all(isfinite, fc[2])) || return nothing
    maximum(abs.(fc[1])) > 10.0 * resid_max && return nothing
    return cand, fc
end

include("topk.jl")

end # module
