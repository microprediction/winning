# Top-k memberships, rank marginals, both Jacobians, and the four
# calibrations. Transliterated from r/winning/R/topk.R (itself a port
# of winning/factor/topk.py, which is the spec); parity/check.jl pins
# every path against the embedded vectors.

clip01(v) = clamp.(v, 0.0, 1.0)

function _resolved_points(lo, hi, sd, points)
    smin = max(minimum(sd), 1e-300)
    # in Float64, and clamp before converting: a narrow enough runner
    # makes the requirement exceed typemax(Int64), and Int(ceil(...))
    # throws InexactError -- in the very regime the cap exists for
    # (#228). Python is unaffected only because its integers are
    # unbounded.
    need = ceil((hi - lo) / (0.5 * smin)) + 1
    if !isfinite(need) || need > 8193
        @warn "top-k lattice cannot resolve the narrowest runner even at 8193 points" smin window=(hi - lo)
    end
    return max(Int(points), _dyadic_points(need))
end

# The adaptive count rounded UP to a dyadic lattice 2^m + 1 (capped at
# 8193), so it is piecewise constant over a factor-of-two band of the
# narrowest scale instead of stepping by one point at every ceil: a count
# moving 611 -> 610 under a 1e-5 relative change of one sd made the
# public map disagree with its scale Jacobian by 100x (#515).
function _dyadic_points(need)
    (!isfinite(need) || need >= 8193) && return 8193
    need <= 2 && return 2
    return Int(min(2.0^ceil(log2(need - 1)) + 1, 8193.0))
end

# searched in STANDARDISED units and mapped back: an absolute 1e-12 floor
# on the widest sd gave a field in units of 1e-18 a window 2e5 times its
# width (#370, as python/R/browser)
function _count_window(mu, sd, k, fn; delta = 1e-12, pad_sds = 2.0)
    m0 = sum(mu) / length(mu)
    c = maximum(sd)
    (isfinite(c) && c > 0) || error("top-k window needs positive scales")
    lo, hi = _count_window_std((mu .- m0) ./ c, sd ./ c, k, fn, delta, pad_sds)
    return (m0 + c * lo, m0 + c * hi)
end

function _count_window_std(mu, sd, k, fn, delta, pad_sds)
    n = length(mu)
    smax = maximum(sd)
    mean_count(x) = sum(1 .- fn((x .- mu) ./ sd).S)
    lo = minimum(mu) - 9 * smax
    step = 9 * smax
    for _ in 1:60
        mean_count(lo) <= delta && break
        lo -= step
        step *= 2
    end
    target_hi = min(k + 2 * log(1 / delta) +
                    sqrt(2 * (k + 1) * log(1 / delta)), n - 1e-4)
    hi = maximum(mu) + 9 * smax
    step = 9 * smax
    for _ in 1:60
        mean_count(hi) >= target_hi && break
        hi += step
        step *= 2
    end
    a, b = lo, hi
    for _ in 1:70
        m = 0.5 * (a + b)
        mean_count(m) < delta ? (a = m) : (b = m)
    end
    xlo = a
    a, b = xlo, hi
    for _ in 1:70
        m = 0.5 * (a + b)
        mean_count(m) < target_hi ? (a = m) : (b = m)
    end
    return (xlo - pad_sds * smax, b + pad_sds * smax)
end

function _topk_grid(mu, sd, k, fn, points; delta = 1e-12)
    lo, hi = _count_window(mu, sd, k, fn; delta = delta)
    # about two points per NARROWEST sd, capped at 8193. The window is
# set by the widest runner and the grid by `points`, so a heterogeneous
# field leaves the narrowest density between samples, and the mass check
# cannot see it: that check is one scalar (memberships sum to k) and
# runner-level errors of opposite sign cancel in it (#224).
    points = _resolved_points(lo, hi, sd, points)
    x = collect(range(lo, hi, length = points))
    z = (x .- mu') ./ sd'                       # (L, n)
    b = fn(z)
    return (x = x, dx = x[2] - x[1], z = z, S = b.S, f = b.f,
            fp = b.fp, F = _lower_cdf(fn, z, b.S), Sc = _count_survival(fn, z, b.S))
end

"""The min-Gumbel's lower-tail CDF, 1 - exp(-exp(c z - gamma)), taken as
-expm1 so it keeps its relative precision where S rounds to one."""
_gumbel_cdf(z) = -expm1.(-exp.(min.(z .* (pi / sqrt(6)) .- EULER, 30)))

"""CDF rows for the count programs. The asymmetric min-Gumbel takes its
own lower tail where S > 1/2: 1 - S rounded a representable last place
to exactly zero (mu = [-35, 0, 0]: 0 for 2.05e-39, #710, as R and
python). Every other base keeps 1 - S, bit for bit."""
function _lower_cdf(fn, z, S)
    S = _unfloored(fn, z, S)
    F = clip01(1 .- S)
    if fn === base_gumbel
        lower = S .> 0.5
        any(lower) && (F[lower] .= clip01(_gumbel_cdf(z[lower])))
    elseif fn === base_normal
        # symmetric: the CDF is the reflected survival, at full relative
        # precision where 1 - S cancels (R's and python's #511)
        lower = S .> 0.5
        any(lower) && (F[lower] .= clip01(_unfloored(fn, -z[lower],
                                                     fn(-z[lower]).S)))
    end
    return F
end

"""S with the normal base's TINY floor replaced by its exact log
survival (R's .unfloored, python's _unfloored, #502): used as a count
probability the floor fabricated rare-place mass. Other bases, and
every point above the floor, are returned as given."""
function _unfloored(fn, z, S)
    fn === base_normal || return S
    low = S .<= TINY
    any(low) || return S
    S = copy(S)
    S[low] .= exp.(max.(_log_sf_normal.(z[low]), -1e300))
    return S
end

"""The survival rows the count programs pair with `_lower_cdf`: S itself
where F was taken from the base's own CDF (so neither is a cancelled
1 - x), otherwise `nothing`, which keeps the old 1 - F. The normal and
Gumbel bases take the error-bounded two-direction deconvolution (#511,
#710); logistic and Laplace keep the one-direction rule."""
_count_survival(fn, z, S) = fn === base_gumbel ? clip01(S) :
                            fn === base_normal ? clip01(_unfloored(fn, z, S)) :
                            nothing

function _count_distribution(F, S = nothing)
    # S, when given, is used in place of 1 - F (accurate where that cancels)
    L, n = size(F)
    C = zeros(L, n + 1)
    C[:, 1] .= 1.0
    for j in 1:n
        f = F[:, j]
        s = S === nothing ? 1 .- f : S[:, j]
        for m in (j + 1):-1:2
            C[:, m] .= C[:, m] .* s .+ C[:, m - 1] .* f
        end
        C[:, 1] .*= s
    end
    return C
end

# The one-direction cavity deconvolution: bases with no lower-tail CDF
# of their own keep it, bit for bit.
function _loo_cdf(C, F, k)
    L, n = size(F)
    out = zeros(n, L)
    for i in 1:n
        Fi = F[:, i]
        Si = 1 .- Fi
        fwd = Si .>= Fi
        s = max.(Si, TINY)
        Q = clip01(C[:, 1] ./ s)
        acc_f = copy(Q)
        if k >= 2
            for m in 2:k
                Q = clip01((C[:, m] .- Fi .* Q) ./ s)
                acc_f .+= Q
            end
        end
        f = max.(Fi, TINY)
        Qb = clip01(C[:, n + 1] ./ f)
        acc_b = copy(Qb)
        if n - 1 >= k + 1
            for mm in (n - 1):-1:(k + 1)
                Qb = clip01((C[:, mm + 1] .- Si .* Qb) ./ f)
                acc_b .+= Qb
            end
        end
        out[i, :] .= clip01(ifelse.(fwd, acc_f, 1 .- acc_b))
    end
    return out
end

function _loo_pmf(C, F, i)
    L, n = size(F)
    Fi = F[:, i]
    Si = 1 .- Fi
    fwd = Si .>= Fi
    s = max.(Si, TINY)
    Qf = zeros(L, n)
    Qf[:, 1] .= clip01(C[:, 1] ./ s)
    for m in 2:n
        Qf[:, m] .= clip01((C[:, m] .- Fi .* Qf[:, m - 1]) ./ s)
    end
    f = max.(Fi, TINY)
    Qb = zeros(L, n)
    Qb[:, n] .= clip01(C[:, n + 1] ./ f)
    for m in (n - 1):-1:1
        Qb[:, m] .= clip01((C[:, m + 1] .- Si .* Qb[:, m + 1]) ./ f)
    end
    Q = copy(Qb)
    Q[fwd, :] .= Qf[fwd, :]
    return Q
end

function _pair_pmf_at(Qi, F, i, k)
    L, n = size(F)
    out = zeros(n, L)
    for j in 1:n
        j == i && continue
        Fj = F[:, j]
        Sj = 1 .- Fj
        fwd = Sj .>= Fj
        s = max.(Sj, TINY)
        Q = clip01(Qi[:, 1] ./ s)
        if k >= 2
            for m in 2:k
                Q = clip01((Qi[:, m] .- Fj .* Q) ./ s)
            end
        end
        f = max.(Fj, TINY)
        Qb = clip01(Qi[:, n] ./ f)
        if n - 1 >= k + 1
            for mm in (n - 1):-1:(k + 1)
                Qb = clip01((Qi[:, mm] .- Sj .* Qb) ./ f)
            end
        end
        out[j, :] .= ifelse.(fwd, Q, Qb)
    end
    return out
end

# Cavity deconvolution with error bounds (#511, #710); mirrors R's
# .count_err/.loo_both/.loo_cdf and python's topk._deconv. Each count
# coefficient carries relative roundoff <= 2 (n+1) eps plus an underflow
# floor; each recursion step adds a subtraction and a division, and the
# running first-order bound tracks the absolute error of every recovered
# coefficient, capped at 1. The old rule ran ONE direction per lattice
# point and integrated the subtraction residue as a rare exact place: a
# Gumbel last place at mu = [-20, 0, 0] came back 1.9e5 times its exact
# 1.05e-22, and at [-12, -4, 4, 12] 5.6e-18 for 1.1e-26.
_count_err(C) = (2 * size(C, 2) + 2) * eps(Float64) .* C .+
                size(C, 2) * floatmin(Float64)

"""Both passes in full, each coefficient from the pass with the smaller
bound: (Q = pmf (L x n), B = the chosen bounds)."""
function _loo_both(C, E, Fi, Si)
    L, n1 = size(C)
    n = n1 - 1
    Q = zeros(L, n)
    B = zeros(L, n)
    inv = 1 ./ max.(Si, 1e-300)
    q = min.(C[:, 1] .* inv, 1.0)
    e = ifelse.(Si .< 1e-300, 1.0, min.(E[:, 1] .* inv, 1.0))
    Q[:, 1] .= q
    B[:, 1] .= e
    for m in 2:n
        q = clip01((C[:, m] .- Fi .* q) .* inv)
        e = min.((E[:, m] .+ Fi .* e) .* inv, 1.0)
        Q[:, m] .= q
        B[:, m] .= e
    end
    inv = 1 ./ max.(Fi, 1e-300)
    q = min.(C[:, n + 1] .* inv, 1.0)
    e = ifelse.(Fi .< 1e-300, 1.0, min.(E[:, n + 1] .* inv, 1.0))
    better = e .< B[:, n]
    Q[better, n] .= q[better]
    B[better, n] .= e[better]
    for m in (n - 1):-1:1
        q = clip01((C[:, m + 1] .- Si .* q) .* inv)
        e = min.((E[:, m + 1] .+ Si .* e) .* inv, 1.0)
        better = e .< B[:, m]
        Q[better, m] .= q[better]
        B[better, m] .= e[better]
    end
    return (Q = Q, B = B)
end

"""P(N_{-i} <= k-1) for every runner, (n, L): the forward head sum or one
minus the backward tail sum, whichever is better bounded; points where
even that bound exceeds 1e-6 of the value are redone per coefficient by
`_loo_both`."""
function _loo_cdf_bounded(C, F, k, S = nothing)
    L, n = size(F)
    k >= n && return ones(n, L)
    S === nothing && (S = 1 .- F)
    E = _count_err(C)
    inv = 1 ./ max.(S, 1e-300)
    q = min.(C[:, 1] .* inv, 1.0)
    e = min.(E[:, 1] .* inv, 1.0)
    e[S .< 1e-300] .= 1.0
    head = copy(q)
    head_e = copy(e)
    for m in 2:k
        q = clip01((C[:, m] .- F .* q) .* inv)
        e = min.((E[:, m] .+ F .* e) .* inv, 1.0)
        head .+= q
        head_e .+= e
    end
    inv = 1 ./ max.(F, 1e-300)
    q = min.(C[:, n + 1] .* inv, 1.0)
    e = min.(E[:, n + 1] .* inv, 1.0)
    e[F .< 1e-300] .= 1.0
    tail = copy(q)
    tail_e = eps(Float64) .+ e
    for m in (n - 1):-1:(k + 1)
        q = clip01((C[:, m + 1] .- S .* q) .* inv)
        e = min.((E[:, m + 1] .+ S .* e) .* inv, 1.0)
        tail .+= q
        tail_e .+= e
    end
    use_head = head_e .<= tail_e
    cdf = ifelse.(use_head, head, 1 .- tail)
    bound = ifelse.(use_head, head_e, tail_e)
    loose = findall(bound .> max.(1e-6 .* cdf, 1e-280))
    if !isempty(loose)
        li = [I[1] for I in loose]
        r = _loo_both(C[li, :], E[li, :], F[loose], S[loose])
        hd = vec(sum(r.Q[:, 1:k], dims = 2))
        he = vec(sum(r.B[:, 1:k], dims = 2))
        tl = vec(sum(r.Q[:, (k + 1):n], dims = 2))
        te = eps(Float64) .+ vec(sum(r.B[:, (k + 1):n], dims = 2))
        cdf[loose] .= ifelse.(he .<= te, hd, 1 .- tl)
    end
    return permutedims(clip01(cdf))
end

"""The leave-one-out count pmf of runner i, (L x n), and its bounds."""
function _loo_pmf_bounded(C, F, i, S = nothing)
    Fi = F[:, i]
    Si = S === nothing ? 1 .- Fi : S[:, i]
    return _loo_both(C, _count_err(C), Fi, Si)
end

"""P(N_{-ij} = k-1) for every j != i: the forward pass up to coefficient
k-1 and the backward pass down to it, each with its running bound
(seeded with the bound on Qi itself), the better-bounded one winning at
each point -- the rule `_loo_both` uses one level up (#511 residual)."""
function _pair_pmf_at_bounded(Qi, F, i, k, S = nothing, Bi = nothing)
    L, n = size(F)
    out = zeros(n, L)
    nn = n - 1                                    # others of i
    k - 1 > nn - 1 && return out                  # only n-2 besides j
    E = _count_err(Qi)
    Bi === nothing || (E = E .+ Bi)
    for j in 1:n
        j == i && continue
        Fj = F[:, j]
        Sj = S === nothing ? 1 .- Fj : S[:, j]
        inv = 1 ./ max.(Sj, 1e-300)
        q = min.(Qi[:, 1] .* inv, 1.0)
        e = ifelse.(Sj .< 1e-300, 1.0, min.(E[:, 1] .* inv, 1.0))
        for m in 2:k
            q = clip01((Qi[:, m] .- Fj .* q) .* inv)
            e = min.((E[:, m] .+ Fj .* e) .* inv, 1.0)
        end
        inv = 1 ./ max.(Fj, 1e-300)
        qb = min.(Qi[:, n] .* inv, 1.0)
        eb = ifelse.(Fj .< 1e-300, 1.0, min.(E[:, n] .* inv, 1.0))
        for cc in (nn - 1):-1:k
            qb = clip01((Qi[:, cc + 1] .- Sj .* qb) .* inv)
            eb = min.((E[:, cc + 1] .+ Sj .* eb) .* inv, 1.0)
        end
        out[j, :] .= ifelse.(e .<= eb, q, qb)
    end
    return out
end

"""The bounded rule where the base supplied accurate survivals `S`, the
one-direction rule (unchanged) where it did not."""
_loo_pmf_either(C, F, i, S) = S === nothing ? (_loo_pmf(C, F, i), nothing) :
                                              _loo_pmf_bounded(C, F, i, S)
_pair_pmf_either(Qi, F, i, k, S, Bi) = S === nothing ?
    _pair_pmf_at(Qi, F, i, k) : _pair_pmf_at_bounded(Qi, F, i, k, S, Bi)

function _topk_with_slopes(mu, sd, k, fn, points)
    g = _topk_grid(mu, sd, k, fn, points)
    C = _count_distribution(g.F, g.Sc)
    cdf = g.Sc === nothing ? _loo_cdf(C, g.F, k) :
          _loo_cdf_bounded(C, g.F, k, g.Sc)
    dens = g.f' ./ sd                            # (n, L)
    dmu = -g.fp' ./ (sd .* sd)
    return (q = vec(sum(dens .* cdf, dims = 2)) .* g.dx,
            slopes = vec(sum(dmu .* cdf, dims = 2)) .* g.dx)
end

function _checked_topk(raw, k, kind; mass_tol = 5e-3)
    t = sum(raw)
    (isfinite(t) && abs(t - k) <= mass_tol * k) ||
        error("$kind captured total membership $t where exactly $k slots exist")
    return clip01(raw .* (k / t))
end

const _TOPK_QMC_NODES = 1024

function _topk_factor_nodes(V, n, qa, caller, D = nothing)
    Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) :
         reshape(Float64.(collect(V)), :, 1)
    r = size(Vm, 2)
    r > 2 && error("$caller is implemented for factor rank <= 2")
    Vm = _gauge_center(Vm)                     # scale-safe (#471)
    if qa === nothing && r > 0
        # qa omitted (the default): the GENERAL RACE'S node rule, as
        # python and the browser. A fixed 15-node Gauss-Hermite rule
        # missed the race by 10.7 points on a sharp rank-one field (#340).
        # Past 1024 nodes a whole 1024-node rule of the same family is
        # used (every node is a full count-program pass): the midpoint
        # grid at rank one, Halton at rank two (python: 2^10 Sobol).
        st = _race_setup(zeros(n), Vm, D === nothing ? ones(n) : D,
                         nothing, nothing, "normal")
        nodes, w = st.F, st.W
        if length(w) > _TOPK_QMC_NODES
            Q = _TOPK_QMC_NODES
            if r == 1
                nodes = reshape(qnorm.(((1:Q) .- 0.5) ./ Q), :, 1)
                w = fill(1.0 / Q, Q)
            else
                hw = halton_normal_nodes(r, Q)
                nodes, w = hw.F, hw.W
            end
        end
        return (Vm = st.V, nodes = nodes, w = w)
    end
    qa === nothing && (qa = 15)
    h = hermite1(qa)
    if r == 0
        # the EMPTY PRODUCT: rank 1 was special-cased and every other
        # rank fell into the rank-2 tensor, so an (n, 0) matrix -- the
        # documented spelling of "no factors" -- was integrated over a
        # two-dimensional factor space it does not have (#309)
        nodes = Matrix{Float64}(undef, 1, 0)
        w = [1.0]
    elseif r == 1
        nodes = reshape(h.nodes, :, 1)
        w = copy(h.weights)
    else
        # first coordinate slowest, matching the reference ordering
        nodes = Matrix{Float64}(undef, qa * qa, 2)
        w = Vector{Float64}(undef, qa * qa)
        t = 0
        for ia in 1:qa, ib in 1:qa
            t += 1
            nodes[t, 1] = h.nodes[ia]
            nodes[t, 2] = h.nodes[ib]
            w[t] = h.weights[ia] * h.weights[ib]
        end
        w ./= sum(w)
    end
    return (Vm = Vm, nodes = nodes, w = w)
end

"""P(runner i among the k smallest), min-wins; slot identity enforced."""
function top_k_probabilities(mu, k; V = nothing, D = nothing,
                             base = "normal", points = 513, qa = nothing)
    mu = Float64.(collect(mu))
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    mu = _translated_home(mu, _as_idio(D, n))     # (#477)
    sd = sqrt.(_as_idio(D, n))
    fn = _base_fn(base)
    V === nothing &&
        return _checked_topk(_topk_with_slopes(mu, sd, k, fn, points).q,
                             k, "top-k race")
    fac = _topk_factor_nodes(V, n, qa, "top_k_probabilities", D)
    raw = zeros(n)
    for q in 1:size(fac.nodes, 1)
        shift = fac.Vm * fac.nodes[q, :]
        raw .+= fac.w[q] .* _topk_with_slopes(mu .+ shift, sd, k, fn, points).q
    end
    return _checked_topk(raw, k, "top-k race")
end

"""Rows of the reflected noise -e: survival 1 - S(-z), density f(-z),
derivative -f'(-z). Used ONLY to place the bottom-k window."""
_reflected(fn) = z -> (b = fn(-z); (S = 1 .- b.S, f = b.f, fp = -b.fp))

"""P(i among the k LARGEST) = int f_i(x) P(M_{-i}(x) <= k-1) dx, with
M_{-i}(x) the number of OTHER runners ABOVE x: the top-k count program
with the survivals in the role the CDFs play there, on the top-k window
of the reflected field."""
function _bottomk_independent(mu, sd, k, fn, points)
    lo_r, hi_r = _count_window(-mu, sd, k, _reflected(fn))
    lo, hi = -hi_r, -lo_r
    points = _resolved_points(lo, hi, sd, points)
    x = collect(range(lo, hi, length = points))
    z = (x .- mu') ./ sd'
    b = fn(z)
    G = clip01(_unfloored(fn, z, b.S))
    # survivals in the role of the CDFs and the CDFs, taken from the
    # base's own lower tail where it has one, in the role of the
    # survivals: 1 - S rounded a Gumbel last place to zero (#710), and
    # the normal's (#511)
    Fc = (fn === base_gumbel || fn === base_normal) ?
         _lower_cdf(fn, z, b.S) : nothing
    C = _count_distribution(G, Fc)
    cdf = Fc === nothing ? _loo_cdf(C, G, k) : _loo_cdf_bounded(C, G, k, Fc)
    dens = b.f' ./ sd
    return vec(sum(dens .* cdf, dims = 2)) .* (x[2] - x[1])
end

"""P(runner i among the k LARGEST), min-wins: the worst k.

Computed DIRECTLY, from the count of runners above x, so a rare
last-place probability keeps its relative precision. It was
`1 - top_k(n - k)`: exact in the continuum, and at mu = [-12, 0, 0] it
returned 0.0 at 513 points and 2.4e-15 at 2049 for a true 7.73e-24
(#365, as python/R/browser)."""
function bottom_k_probabilities(mu, k; V = nothing, D = nothing,
                                base = "normal", points = 513, qa = nothing)
    mu = Float64.(collect(mu))
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    mu = _translated_home(mu, _as_idio(D, n))     # (#477)
    sd = sqrt.(_as_idio(D, n))
    fn = _base_fn(base)
    V === nothing &&
        return _checked_topk(_bottomk_independent(mu, sd, k, fn, points),
                             k, "bottom-k race")
    fac = _topk_factor_nodes(V, n, qa, "bottom_k_probabilities", D)
    raw = zeros(n)
    for q in 1:size(fac.nodes, 1)
        raw .+= fac.w[q] .* _bottomk_independent(
            mu .+ fac.Vm * fac.nodes[q, :], sd, k, fn, points)
    end
    return _checked_topk(raw, k, "bottom-k race")
end

"""Both top-k Jacobians (dq/dmu, dq/dsigma); factor rank <= 2 via the
exact Gauss-Hermite node mixture."""
function top_k_jacobians(mu, k; D = nothing, base = "normal",
                         points = 513, V = nothing, qa = nothing)
    mu = Float64.(collect(mu))
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    mu = _translated_home(mu, _as_idio(D, n))     # (#477)
    if V !== nothing
        fac = _topk_factor_nodes(V, n, qa, "top_k_jacobians", D)
        Jm = zeros(n, n)
        Js = zeros(n, n)
        for q in 1:size(fac.nodes, 1)
            node = top_k_jacobians(mu .+ fac.Vm * fac.nodes[q, :], k;
                                   D = D, base = base, points = points)
            Jm .+= fac.w[q] .* node.Jmu
            Js .+= fac.w[q] .* node.Jsigma
        end
        return (Jmu = Jm, Jsigma = Js)
    end
    Dv = _as_idio(D, n)
    sd = sqrt.(Dv)
    fn = _base_fn(base)
    g = _topk_grid(mu, sd, k, fn, points)
    dens = g.f' ./ sd                            # (n, L)
    zdens = (g.z .* g.f)' ./ sd
    C = _count_distribution(g.F, g.Sc)
    Jm = zeros(n, n)
    Js = zeros(n, n)
    for i in 1:n
        Qi, Bi = _loo_pmf_either(C, g.F, i, g.Sc)
        pair = _pair_pmf_either(Qi, g.F, i, k, g.Sc, Bi)
        kern = pair .* dens[i, :]'
        row_mu = vec(sum(kern .* dens, dims = 2)) .* g.dx
        row_sd = vec(sum(kern .* zdens, dims = 2)) .* g.dx
        row_mu[i] = 0.0
        row_mu[i] = -sum(row_mu)
        cdf_i = vec(sum(Qi[:, 1:k], dims = 2))
        dfdsd = -(g.z[:, i] .* g.fp[:, i] .+ g.f[:, i]) ./ Dv[i]
        row_sd[i] = sum(dfdsd .* cdf_i) * g.dx
        Jm[i, :] .= row_mu
        Js[i, :] .= row_sd
    end
    return (Jmu = Jm, Jsigma = Js)
end

function _rank_matrix_node(mu, sd, fn, points)
    n = length(mu)
    g = _topk_grid(mu, sd, n - 1, fn, points)
    C = _count_distribution(g.F, g.Sc)
    P = zeros(n, n)
    for i in 1:n
        Qi, _ = _loo_pmf_either(C, g.F, i, g.Sc)
        P[i, :] .= vec(sum(Qi .* (g.f[:, i] ./ sd[i]), dims = 1)) .* g.dx
    end
    return P
end

"""The full rank marginals: entry [i, r] = P(runner i finishes r-th).

`V` (factor rank <= 2) and `qa` price a factor-correlated race by mixing
the conditional rank matrices over the same Gauss-Hermite factor nodes
top_k_probabilities uses, as the Python reference does. This port used
to have no `V` at all, so the correlated call was a keyword error while
the adjacent top-k verbs priced it (#202). Column 1 is correlated top-1
membership and cumulative row sums are correlated top-k."""
function rank_probabilities(mu; D = nothing, base = "normal", points = 513,
                            V = nothing, qa = nothing)
    mu = Float64.(collect(mu))
    n = length(mu)
    mu = _translated_home(mu, _as_idio(D, n))     # (#477)
    sd = sqrt.(_as_idio(D, n))
    fn = _base_fn(base)
    P = if V === nothing
        _rank_matrix_node(mu, sd, fn, points)
    else
        fac = _topk_factor_nodes(V, n, qa, "rank_probabilities", D)
        acc = zeros(n, n)
        for q in 1:size(fac.nodes, 1)
            acc .+= fac.w[q] .* _rank_matrix_node(
                mu .+ fac.Vm * fac.nodes[q, :], sd, fn, points)
        end
        acc
    end
    # Check what is RETURNED. Row normalisation makes every row exact and
    # MOVES the columns, so a matrix that passed the raw check could fail
    # the stated identity afterwards with nothing looking (#203). The
    # columns are not forced: alternating scaling would make both exact by
    # hiding the under-resolution that caused it, and the same field at
    # 2001 points has a column error of 3.5e-9.
    # BOTH checks: independent, and mistaken for alternatives once. Row
    # normalisation can ERASE a gross raw defect and leave the result just
    # inside tolerance (#221).
    re_raw = maximum(abs.(vec(sum(P, dims = 2)) .- 1))
    ce_raw = maximum(abs.(vec(sum(P, dims = 1)) .- 1))
    (all(isfinite, P) && re_raw <= 5e-3 && ce_raw <= 5e-3) ||
        error("rank marginals defective before normalisation: row-sum " *
              "error $re_raw, column-sum error $ce_raw. Raise points=.")
    P = clip01(P ./ max.(vec(sum(P, dims = 2)), 1e-300))
    re = maximum(abs.(vec(sum(P, dims = 2)) .- 1))
    ce = maximum(abs.(vec(sum(P, dims = 1)) .- 1))
    (all(isfinite, P) && re <= 5e-3 && ce <= 5e-3) ||
        error("rank marginals defective in the RETURNED matrix after row " *
              "normalisation: row-sum error $re, column-sum error $ce. " *
              "Raise points=.")
    return P
end

# A relative target as a law of total `mass`, scale-safe as python's
# winning.shapes.rescaled_target: the raw sum overflows for finite
# [1e308, ...] and mass / sum overflows for a finite sum below
# mass / floatmax (4e-309 .* q), and either collapsed the target before
# any solver ran (#461, #483). Ordinary inputs take the plain expression.
function _rescaled_target(t, mass = 1.0)
    t = Float64.(collect(t))
    s = sum(t)
    if !isfinite(s)
        t = t ./ maximum(t)
        s = sum(t)
    end
    f = mass / s
    return isfinite(f) ? t .* f : (t ./ s) .* mass
end

function _validated_topk_target(q, k, n, target_floor)
    target = Float64.(collect(q))
    length(target) == n || error("target has $(length(target)) entries for $n runners")
    floored = falses(n)
    if target_floor !== nothing
        (isfinite(target_floor) && target_floor > 0) ||
            error("target_floor must be a finite positive number")
        # a membership floor: applied to memberships normalized to k
        # slots, not to the caller's units (#592)
        all(isfinite, target) && all(>=(0), target) ||
            error("target entries must be finite and non-negative")
        target = target ./ maximum(target)
        target .*= k / sum(target)
        # in the FINAL k-slot scale: floored entries sit exactly at the
        # floor, the others give up that mass proportionally (python
        # races._apply_floor); clamp-then-renormalise scaled the floor
        # back down to 0.167 for [.7, .3, 0] at .2 (#592)
        floored = target .< target_floor
        scaled = copy(target)
        for _ in 1:n
            free = k - target_floor * count(floored)
            rest = sum(target[.!floored])
            (free > 0 && rest > 0) ||
                error("target_floor=$target_floor is infeasible: flooring $(count(floored)) of $n entries at it leaves no mass for the rest of a target of total $k")
            scaled = ifelse.(floored, target_floor, target .* (free / rest))
            newly = .!floored .& (scaled .< target_floor)
            any(newly) || break
            floored = floored .| newly
        end
        target = scaled
    elseif any(target .<= 0)
        error("all target memberships must be positive")
    end
    target = _rescaled_target(target, k)
    any(target .>= 1) &&
        error("after renormalizing to k slots, a target membership is >= 1")
    return (target = target, floored = floored)
end

"""Invert one top-k curve for mean-zero locations (logit residuals)."""
function abilities_from_topk(q, k; V = nothing, D = nothing,
                             base = "normal", points = 513, qa = nothing,
                             n_iter = 80, tol = 1e-8,
                             target_floor = nothing, return_info = false)
    n = length(q)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    vt = _validated_topk_target(q, k, n, target_floor)
    target = vt.target
    sd = sqrt.(_as_idio(D, n))
    fn = _base_fn(base)
    fac = V === nothing ? nothing :
          _topk_factor_nodes(V, n, qa, "abilities_from_topk", D)
    logit_t = log.(target) .- log1p.(-target)
    logt = log.(target)
    # in the field's own units, as python and R (#100): a dimensionless
    # start, absolute step caps and an absolute slope floor left
    # abilities_from_topk 0.46 off its target at sd 0.1 and 0.39 at
    # sd 1e3, where the same race at sd 1 converged
    sv = fac === nothing ? 0.0 : sum(fac.Vm .^ 2) / n
    Dn = sort(sd .^ 2)
    medD = isodd(n) ? Dn[(n + 1) ÷ 2] : 0.5 * (Dn[n ÷ 2] + Dn[n ÷ 2 + 1])
    scale = sqrt(medD + sv)
    mu = -(logt .- sum(logt) / n) ./ 2 .* scale
    # damping, as python: the k = 1 gate is the win race's top-two share
    # (two runners holding nearly all the win mass two-cycle undamped at
    # any n -- `n > 2 ? 1.0 : 0.7` failed #151's fixtures at n = 3, 10,
    # 30, #408), and the sweeps adapt it to the contraction they observe
    top2 = (k == 1 && n > 2) ? sum(sort(target)[(end - 1):end]) : 1.0
    alpha = (n == 2 || (k == 1 && top2 > 0.8)) ? 0.7 : 1.0
    function fwd(m)
        local qraw, sl
        if fac === nothing
            ws = _topk_with_slopes(m, sd, k, fn, points)
            qraw, sl = ws.q, ws.slopes
        else
            qraw = zeros(n)
            sl = zeros(n)
            for j in 1:size(fac.nodes, 1)
                ws = _topk_with_slopes(m .+ fac.Vm * fac.nodes[j, :],
                                       sd, k, fn, points)
                qraw .+= fac.w[j] .* ws.q
                sl .+= fac.w[j] .* ws.slopes
            end
        end
        qhat = _checked_topk(qraw, k, "top-k inversion")
        resid = (log.(max.(qhat, TINY)) .- log.(max.(1 .- qhat, TINY))) .- logit_t
        dlogit = min.(sl ./ max.(qhat .* (1 .- qhat), TINY), -1e-6 / scale)
        return resid, dlogit
    end
    mu, _, resid_max, iters = _jacobi_sweeps(mu, fwd, scale, alpha, n_iter,
                                             tol)
    converged = resid_max < tol
    return_info && return (mu = mu, converged = converged,
                           max_logit_residual = resid_max,
                           iterations = iters, floored = vt.floored)
    converged || @warn "abilities_from_topk did not converge" resid_max iters
    return mu
end

"""Joint (loc, scale) from two membership curves; refuses fixed
loadings (two curves are then one equation short)."""
# Bounded deterministic retry for a failed inverse (#779, #781), shared by
# abilities_from_rank_marginal and loc_scale_from_topk_pair as in python. A
# first attempt that converged is returned untouched (bitwise, no extra
# cost); otherwise attempt(start) runs from each of the fixed starts(first)
# in order and the first converged outcome wins, else the attempt with the
# smallest getproperty(out, key) is returned, still converged = false.
# iterations becomes the total over every attempt. A retry that throws is
# skipped: it must not discard the completed attempts (#855).
function _retry_on_failure(first, attempt, starts, key)
    # n_iter = 0 prices the start and performs no update: never retried
    (first.converged || first.iterations == 0) && return first
    best = first
    total = first.iterations
    for start in starts(first)
        out = try
            attempt(start)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        out === nothing && continue
        total += out.iterations
        if out.converged
            best = out
            break
        end
        getproperty(out, key) < getproperty(best, key) && (best = out)
    end
    return merge(best, (iterations = total,))
end

# Acklam's rational inverse normal CDF, the browser's invNormalRational:
# plain arithmetic, so every port computes the same retry start.
function _probit_rational(p::Float64)
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
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
    p > 1 - pl && return -_probit_rational(1 - p)
    q = p - 0.5
    r2 = q * q
    return (((((a[1]*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*r2+a[6])*q /
           (((((b[1]*r2+b[2])*r2+b[3])*r2+b[4])*r2+b[5])*r2+1)
end

# The retry start of the cold loc/scale solve (#781), as python's
# _loc_scale_starts: with the depth thresholds held fixed, the gap between
# a runner's deeper and shallower probits is proportional to 1/sigma_i.
# Geometric-mean-one, clipped to the solver's log-scale box, as a D0.
function _loc_scale_starts(q1, k1, q2, k2)
    n = length(q1)
    lo, klo, hi, khi = k1 < k2 ? (q1, k1, q2, k2) : (q2, k2, q1, k1)
    tlo = _validated_topk_target(lo, klo, n, nothing).target
    thi = _validated_topk_target(hi, khi, n, nothing).target
    ls = [-log(max(_probit_rational(thi[i]) - _probit_rational(tlo[i]),
                   1e-3)) for i in 1:n]
    m = sum(ls) / n
    return [exp.(2 .* clamp.(ls .- m, -3, 3))]
end

function loc_scale_from_topk_pair(q1, k1, q2, k2; D0 = nothing,
                                  base = "normal", points = 513,
                                  n_iter = 60, tol = 1e-8, ridge = 0.0,
                                  mu0 = nothing, V = nothing,
                                  return_info = false)
    # the cold start (no D0, no mu0) can stall far from an exact board;
    # only then, one retry from a scale-informed D0 (#781), as python
    once(d, m) = _loc_scale_once(q1, k1, q2, k2, d, base, points, n_iter,
                                 tol, ridge, m, V)
    out = once(D0, mu0)
    if D0 === nothing && mu0 === nothing
        out = _retry_on_failure(out, d -> once(d, nothing),
                                f -> _loc_scale_starts(q1, k1, q2, k2),
                                :max_logit_residual)
    end
    return_info && return out
    out.converged || @warn "loc_scale_from_topk_pair did not converge" resid_max = out.max_logit_residual
    return (mu = out.mu, sd = out.sd)
end

function _loc_scale_once(q1, k1, q2, k2, D0, base, points, n_iter, tol,
                         ridge, mu0, V)
    V === nothing || error(
        "loc_scale_from_topk_pair with fixed factor loadings is " *
        "under-identified: use abilities_from_topk(V=) at fixed scales")
    n = length(q1)
    k1, k2 = Int(k1), Int(k2)
    k1 == k2 && error("k1 == k2 gives one curve twice")
    for kk in (k1, k2)
        1 <= kk <= n - 1 || error("k must be in [1, n-1]; got k=$kk, n=$n")
    end
    t1 = _validated_topk_target(q1, k1, n, nothing).target
    t2 = _validated_topk_target(q2, k2, n, nothing).target
    lt1 = log.(t1) .- log1p.(-t1)
    lt2 = log.(t2) .- log1p.(-t2)
    sd = sqrt.(_as_idio(D0, n; name = "D0"))
    local mu
    if mu0 !== nothing
        mu = Float64.(collect(mu0)) .- sum(mu0) / n
    else
        ka, ta = k1 < k2 ? (k1, t1) : (k2, t2)
        # warm start only: the LM loop refines, loose tolerance by design
        mu = abilities_from_topk(ta, ka; D = sd .^ 2, base = base,
                                 points = points, n_iter = 20, tol = 1e-3,
                                 return_info = true).mu
    end
    # the RETURN gauge (mean-zero mu, geometric-mean-one sd) applied to
    # the start as well: it was applied only to accepted LM steps, so an
    # exact warm start converged before any step and came back in the
    # caller's units -- sd [2, 3, 4, 5] where python returns
    # [0.60, 0.91, 1.21, 1.51] (#595, julia's copy of #360). Ranks are
    # invariant to it, so no probability moves.
    c0 = exp(sum(log.(sd)) / n)
    mu = (mu .- sum(mu) / n) ./ c0
    sd = sd ./ c0
    sqr = sqrt(max(ridge, 0.0))
    function logits(m, s)
        qh1 = clamp.(top_k_probabilities(m, k1; D = s .^ 2, base = base,
                                         points = points), TINY, 1 - 1e-15)
        qh2 = clamp.(top_k_probabilities(m, k2; D = s .^ 2, base = base,
                                         points = points), TINY, 1 - 1e-15)
        r = vcat(log.(qh1) .- log1p.(-qh1) .- lt1,
                 log.(qh2) .- log1p.(-qh2) .- lt2,
                 sqr .* log.(s))
        return (r = r, qh1 = qh1, qh2 = qh2)
    end
    lg = logits(mu, sd)
    r, qh1, qh2 = lg.r, lg.qh1, lg.qh2
    cost = sum(abs2, r)
    resid_max = maximum(abs.(r[1:(2n)]))
    lam = 1e-6
    iters = 0
    last_accepted = true
    last_grad = Inf
    for it in 1:n_iter
        iters = it
        resid_max < tol && break
        blocks = Matrix{Float64}[]
        for (kk, qh) in ((k1, qh1), (k2, qh2))
            Jp = top_k_jacobians(mu, kk; D = sd .^ 2, base = base,
                                 points = points)
            g = 1 ./ max.(qh .* (1 .- qh), TINY)
            push!(blocks, hcat(Jp.Jmu .* g, (Jp.Jsigma .* sd') .* g))
        end
        push!(blocks, hcat(zeros(n, n), sqr .* _I(n)))
        J = vcat(blocks...)
        JtJ = J' * J
        Jtr = J' * r
        last_grad = maximum(abs.(Jtr))
        accepted = false
        for _ in 1:8
            step = try
                (JtJ + lam .* _I(2n)) \ (-Jtr)
            catch
                lam *= 8
                continue
            end
            mu_n = mu .+ step[1:n]
            ls_n = clamp.(log.(sd) .+ step[(n + 1):(2n)], -3, 3)
            cc = exp(sum(ls_n) / n)
            sd_n = exp.(ls_n .- sum(ls_n) / n)
            mu_n = (mu_n .- sum(mu_n) / n) ./ cc
            lg_n = try
                logits(mu_n, sd_n)
            catch
                lam *= 8
                continue
            end
            cost_n = sum(abs2, lg_n.r)
            if cost_n < cost
                mu, sd = mu_n, sd_n
                r, qh1, qh2 = lg_n.r, lg_n.qh1, lg_n.qh2
                cost = cost_n
                resid_max = maximum(abs.(r[1:(2n)]))
                lam = max(lam / 3, 1e-10)
                accepted = true
                break
            end
            lam *= 8
        end
        last_accepted = accepted
        accepted || break
    end
    # A ridge stall counts only where it is a stationary point of the
    # penalized objective AND the board passes the exact nesting
    # necessity P(top k1) <= P(top k2) for k1 < k2. `resid < tol ||
    # (ridge > 0 && stalled)` certified every rejected step, including
    # q1 = [.8, .1, .1] against q2 = [.2, .9, .9], which no (mu, sigma)
    # can fit: converged = true at a logit residual of 1.57 and a 0.35
    # repricing miss (#353, as python/R/browser).
    lo_t, hi_t = k1 < k2 ? (t1, t2) : (t2, t1)
    nested = all(lo_t .<= hi_t .+ 1e-12)
    fit_ok = resid_max < tol
    stationary = fit_ok || (!last_accepted && iters > 0 &&
                            last_grad <= 1e-6 * max(1.0, cost))
    converged = fit_ok || (sqr > 0 && stationary && nested)
    return (mu = mu, sd = sd, converged = converged,
            max_logit_residual = resid_max,
            iterations = iters, fit_converged = fit_ok,
            stationary = stationary, nested = nested)
end

_I(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"""Win plus EXACTLY-second marginals: P(2nd) + P(win) = P(top-2)."""
function loc_scale_from_win_and_second(p_win, p_second; kwargs...)
    n = length(p_win)
    length(p_second) == n || error("p_win and p_second must have equal length")
    (any(p_win .<= 0) || any(p_second .<= 0)) &&
        error("all win and second probabilities must be positive")
    p1 = _rescaled_target(p_win)
    top2 = p1 .+ _rescaled_target(p_second)
    return loc_scale_from_topk_pair(p1, 1, top2, 2; kwargs...)
end

function _rank_marginal_with_jacobian(mu, sd, r, fn, points)
    n = length(mu)
    g = _topk_grid(mu, sd, n - 1, fn, points)
    dens = g.f' ./ sd                            # (n, L)
    C = _count_distribution(g.F, g.Sc)
    p = zeros(n)
    J = zeros(n, n)
    for i in 1:n
        Qi, Bi = _loo_pmf_either(C, g.F, i, g.Sc)
        p[i] = sum(Qi[:, r] .* dens[i, :]) * g.dx
        # P(N_{-ij} = n-1) is identically zero (only n-2 others exist)
        pair = r <= n - 1 ? _pair_pmf_either(Qi, g.F, i, r, g.Sc, Bi) :
               zeros(n, length(g.x))
        if r >= 2
            pair = pair .- _pair_pmf_either(Qi, g.F, i, r - 1, g.Sc, Bi)
        end
        kern = pair .* dens[i, :]'
        row = vec(sum(kern .* dens, dims = 2)) .* g.dx
        row[i] = 0.0
        row[i] = -sum(row)
        J[i, :] .= row
    end
    return (p = p, J = J)
end

# The retry starts of abilities_from_rank_marginal (#779), as python's
# _rank_marginal_starts: the original start nudged -/+ 0.3 typical scale
# along the stalled log residual, then the failed trajectory reflected
# through the start.
function _rank_marginal_starts(mu0, mu_stall, resid, D)
    n = length(mu_stall)
    m0 = mu0 === nothing ? zeros(n) : Float64.(collect(mu0)) .- sum(mu0) / n
    u = resid .- sum(resid) / n
    u = u ./ max(maximum(abs.(u)), 1e-300)
    h = 0.3 * sqrt(sum(_as_idio(D, n)) / n)
    return [m0 .- h .* u, m0 .+ h .* u, 2 .* m0 .- mu_stall]
end

"""Invert one EXACT-rank marginal at frozen scales (two-branched for
r >= 2; mu0 selects the branch). One damped trajectory can stall at a
positive-residual stationary point; only then, a bounded deterministic
retry (#779)."""
function abilities_from_rank_marginal(p, r; mu0 = nothing, D = nothing,
                                      base = "normal", points = 513,
                                      n_iter = 60, tol = 1e-8,
                                      return_info = false)
    once(m) = _rank_marginal_once(p, r, m, D, base, points, n_iter, tol)
    out = _retry_on_failure(once(mu0), once,
                            f -> _rank_marginal_starts(mu0, f.mu, f.resid, D),
                            :max_log_residual)
    return_info && return (mu = out.mu, converged = out.converged,
                           max_log_residual = out.max_log_residual,
                           iterations = out.iterations)
    out.converged || @warn "abilities_from_rank_marginal did not converge" resid_max = out.max_log_residual
    return out.mu
end

function _rank_marginal_once(p, r, mu0, D, base, points, n_iter, tol)
    n = length(p)
    # a whole-number rank with a descriptive refusal (#317; Int(1.5)
    # threw an InexactError naming nothing the caller passed)
    (r isa Real && !(r isa Bool) && isfinite(r) && r == floor(r)) ||
        throw(ArgumentError("r must be a whole-number rank; got $r"))
    r = Int(r)
    1 <= r <= n || error("rank must be in [1, n]; got r=$r, n=$n")
    all(isfinite, p) || error("rank probabilities must be finite")
    any(p .<= 0) && error("all rank probabilities must be positive")
    logt = log.(_rescaled_target(p))
    sd = sqrt.(_as_idio(D, n))
    fn = _base_fn(base)
    zz = [-2.3, -1.1, -0.35, 0.6, 1.7]
    symmetric = all(abs.(fn(zz).S .+ fn(-zz).S .- 1) .< 1e-12)
    if mu0 === nothing && isodd(n) && r == (n + 1) ÷ 2 && symmetric
        # the exact middle rank of an odd field under a symmetric base is
        # even in mu: zero Jacobian at the zero start (#378)
        throw(ArgumentError("the exact middle rank r=$r of an odd field " *
            "(n=$n) under a symmetric base is unchanged by mu -> -mu, so " *
            "the zero start is stationary; pass mu0 to choose the branch"))
    end
    mu = mu0 === nothing ? zeros(n) : Float64.(collect(mu0)) .- sum(mu0) / n
    st = _rank_marginal_with_jacobian(mu, sd, r, fn, points)
    resid = log.(max.(st.p, TINY)) .- logt
    cost = sum(abs2, resid)
    resid_max = maximum(abs.(resid))
    lam = 1e-6
    iters = 0
    for it in 1:n_iter
        iters = it
        resid_max < tol && break
        Jlog = st.J ./ max.(st.p, TINY)
        A = Jlog' * Jlog
        gvec = Jlog' * resid
        accepted = false
        for _ in 1:8
            step = try
                (A + lam * _I(n)) \ (-gvec)
            catch
                lam *= 8
                continue
            end
            mu_n = mu .+ step
            mu_n .-= sum(mu_n) / n
            st_n = _rank_marginal_with_jacobian(mu_n, sd, r, fn, points)
            r_n = log.(max.(st_n.p, TINY)) .- logt
            cost_n = sum(abs2, r_n)
            if cost_n < cost
                mu, st, resid, cost = mu_n, st_n, r_n, cost_n
                resid_max = maximum(abs.(resid))
                lam = max(lam / 3, 1e-10)
                accepted = true
                break
            end
            lam *= 8
        end
        accepted || break
    end
    converged = resid_max < tol
    return (mu = mu, resid = resid, converged = converged,
            max_log_residual = resid_max, iterations = iters)
end
