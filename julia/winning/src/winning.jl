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

function ndtr(z::Float64)
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
