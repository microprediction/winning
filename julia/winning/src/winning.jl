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
    S = max.(1 .- ndtr.(z), TINY)
    f = npdf.(z)
    return (S = S, f = f, fp = -z .* f)
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

"""Pruned Gauss-Hermite product rule, first coordinate slowest, pruned
WITHOUT renormalizing (matching the reference exactly)."""
function hermite_nodes(k::Int, order::Int = 15, prune::Float64 = 1e-7)
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
    return (F = F[keep, :], W = W[keep])
end

function halton_normal_nodes(r::Int, n::Int)
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
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

function _race_setup(mu, V, D, F, W, base)
    mu = Float64.(collect(mu))
    n = length(mu)
    D = D === nothing ? ones(n) : Float64.(collect(D))
    if V === nothing
        Vm = zeros(n, 1)
        Fm = zeros(1, 1)
        Wv = [1.0]
    else
        Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) : reshape(Float64.(collect(V)), :, 1)
        Vm = Vm .- sum(Vm, dims = 1) ./ n      # common column is gauge
        if F === nothing || W === nothing
            sharp = sqrt(2) * maximum(sqrt.(vec(sum(Vm .^ 2, dims = 2))) ./
                                      sqrt.(max.(D, TINY)))
            r = size(Vm, 2)
            if r >= 2 && sharp > 3.0
                hw = halton_normal_nodes(r, 2^13)
                Fm, Wv = hw.F, hw.W
            elseif r == 1 && ceil(8 * sharp) > 201
                Q = Int(min(ceil(8 * sharp), 4001))
                Fm = reshape(qnorm.(((1:Q) .- 0.5) ./ Q), :, 1)
                Wv = fill(1.0 / Q, Q)
            elseif 15.0^r > 1e5
                hw = halton_normal_nodes(r, 2^13)
                Fm, Wv = hw.F, hw.W
            else
                cap = r == 1 ? 201 : r == 2 ? 41 : 15
                Q = Int(min(max(ceil(8 * sharp), 15), cap))
                hw = hermite_nodes(size(Vm, 2), Q)
                Fm, Wv = hw.F, hw.W
            end
        else
            Fm = F isa AbstractMatrix ? Float64.(Matrix(F)) : reshape(Float64.(collect(F)), :, 1)
            Wv = Float64.(collect(W))
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
    st = _race_setup(mu, V, D, F, W, base)
    n = length(st.mu)
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
        logS = log.(b.S)
        Lfield = vec(sum(logS, dims = 1))        # per lattice point
        rest = exp.(clamp.(Lfield' .- logS, -745, 0))
        p .+= st.W[q] .* vec(sum(f .* rest, dims = 2)) .* dx
        slope .+= st.W[q] .* vec(sum(-b.fp ./ sd .^ 2 .* rest, dims = 2)) .* dx
    end
    total = sum(p)
    return_slopes && return (p = p ./ total, slopes = slope ./ total)
    return p ./ total
end

"""Invert the general race: mean-zero mu with matching probabilities,
by the damped own-slope iteration with residual-proportional caps."""
function abilities_from_race(p; V = nothing, D = nothing, F = nothing,
                             W = nothing, base = "normal", points = 257,
                             n_iter = 60, tol = 1e-8)
    target = Float64.(collect(p))
    any(target .<= 0) && error("all target probabilities must be positive")
    target ./= sum(target)
    logt = log.(target)
    mu = -(logt .- sum(logt) / length(logt)) ./ 2
    alpha = length(target) > 2 ? 1.0 : 0.7
    for _ in 1:n_iter
        ps = race_probabilities(mu; V = V, D = D, F = F, W = W,
                                base = base, points = points,
                                return_slopes = true)
        resid = log.(max.(ps.p, TINY)) .- logt
        maximum(abs.(resid)) < tol && break
        dlogp = min.(ps.slopes ./ max.(ps.p, TINY), -1e-6)
        lim = min.(2, 10 .* abs.(resid))
        mu .-= clamp.(alpha .* resid ./ dlogp, -lim, lim)
        mu .-= sum(mu) / length(mu)
    end
    return mu
end

include("topk.jl")

end # module
