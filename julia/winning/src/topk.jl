# Top-k memberships, rank marginals, both Jacobians, and the four
# calibrations. Transliterated from r/winning/R/topk.R (itself a port
# of winning/factor/topk.py, which is the spec); parity/check.jl pins
# every path against the embedded vectors.

clip01(v) = clamp.(v, 0.0, 1.0)

function _count_window(mu, sd, k, fn; delta = 1e-12, pad_sds = 2.0)
    n = length(mu)
    smax = max(maximum(sd), 1e-12)
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
    x = collect(range(lo, hi, length = points))
    z = (x .- mu') ./ sd'                       # (L, n)
    b = fn(z)
    return (x = x, dx = x[2] - x[1], z = z, S = b.S, f = b.f,
            fp = b.fp, F = clip01(1 .- b.S))
end

function _count_distribution(F)
    L, n = size(F)
    C = zeros(L, n + 1)
    C[:, 1] .= 1.0
    for j in 1:n
        f = F[:, j]
        for m in (j + 1):-1:2
            C[:, m] .= C[:, m] .* (1 .- f) .+ C[:, m - 1] .* f
        end
        C[:, 1] .*= 1 .- f
    end
    return C
end

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

function _topk_with_slopes(mu, sd, k, fn, points)
    g = _topk_grid(mu, sd, k, fn, points)
    C = _count_distribution(g.F)
    cdf = _loo_cdf(C, g.F, k)
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

function _topk_factor_nodes(V, n, qa, caller)
    Vm = V isa AbstractMatrix ? Float64.(Matrix(V)) :
         reshape(Float64.(collect(V)), :, 1)
    r = size(Vm, 2)
    r > 2 && error("$caller is implemented for factor rank <= 2")
    Vm = Vm .- sum(Vm, dims = 1) ./ n
    h = hermite1(qa)
    if r == 1
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
                             base = "normal", points = 513, qa = 15)
    mu = Float64.(collect(mu))
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    sd = sqrt.(D === nothing ? ones(n) : Float64.(collect(D)))
    fn = _base_fn(base)
    V === nothing &&
        return _checked_topk(_topk_with_slopes(mu, sd, k, fn, points).q,
                             k, "top-k race")
    fac = _topk_factor_nodes(V, n, qa, "top_k_probabilities")
    raw = zeros(n)
    for q in 1:size(fac.nodes, 1)
        shift = fac.Vm * fac.nodes[q, :]
        raw .+= fac.w[q] .* _topk_with_slopes(mu .+ shift, sd, k, fn, points).q
    end
    return _checked_topk(raw, k, "top-k race")
end

function bottom_k_probabilities(mu, k; kwargs...)
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    return 1 .- top_k_probabilities(mu, n - k; kwargs...)
end

"""Both top-k Jacobians (dq/dmu, dq/dsigma); factor rank <= 2 via the
exact Gauss-Hermite node mixture."""
function top_k_jacobians(mu, k; D = nothing, base = "normal",
                         points = 513, V = nothing, qa = 15)
    mu = Float64.(collect(mu))
    n = length(mu)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    if V !== nothing
        fac = _topk_factor_nodes(V, n, qa, "top_k_jacobians")
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
    Dv = D === nothing ? ones(n) : Float64.(collect(D))
    sd = sqrt.(Dv)
    fn = _base_fn(base)
    g = _topk_grid(mu, sd, k, fn, points)
    dens = g.f' ./ sd                            # (n, L)
    zdens = (g.z .* g.f)' ./ sd
    C = _count_distribution(g.F)
    Jm = zeros(n, n)
    Js = zeros(n, n)
    for i in 1:n
        Qi = _loo_pmf(C, g.F, i)
        pair = _pair_pmf_at(Qi, g.F, i, k)
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

"""The full rank marginals: entry [i, r] = P(runner i finishes r-th)."""
function rank_probabilities(mu; D = nothing, base = "normal", points = 513)
    mu = Float64.(collect(mu))
    n = length(mu)
    sd = sqrt.(D === nothing ? ones(n) : Float64.(collect(D)))
    fn = _base_fn(base)
    g = _topk_grid(mu, sd, n - 1, fn, points)
    C = _count_distribution(g.F)
    P = zeros(n, n)
    for i in 1:n
        Qi = _loo_pmf(C, g.F, i)
        P[i, :] .= vec(sum(Qi .* (g.f[:, i] ./ sd[i]), dims = 1)) .* g.dx
    end
    rows = vec(sum(P, dims = 2))
    cols = vec(sum(P, dims = 1))
    (all(isfinite, P) && maximum(abs.(rows .- 1)) <= 5e-3 &&
     maximum(abs.(cols .- 1)) <= 5e-3) ||
        error("rank marginals defective; raise points=")
    return clip01(P ./ rows)
end

function _validated_topk_target(q, k, n, target_floor)
    target = Float64.(collect(q))
    length(target) == n || error("target has $(length(target)) entries for $n runners")
    floored = falses(n)
    if target_floor !== nothing
        target_floor > 0 || error("target_floor must be positive")
        floored = target .< target_floor
        target = max.(target, target_floor)
    elseif any(target .<= 0)
        error("all target memberships must be positive")
    end
    target .*= k / sum(target)
    any(target .>= 1) &&
        error("after renormalizing to k slots, a target membership is >= 1")
    return (target = target, floored = floored)
end

"""Invert one top-k curve for mean-zero locations (logit residuals)."""
function abilities_from_topk(q, k; V = nothing, D = nothing,
                             base = "normal", points = 513, qa = 15,
                             n_iter = 80, tol = 1e-8,
                             target_floor = nothing, return_info = false)
    n = length(q)
    k = Int(k)
    1 <= k <= n - 1 || error("k must be in [1, n-1]; got k=$k, n=$n")
    vt = _validated_topk_target(q, k, n, target_floor)
    target = vt.target
    sd = sqrt.(D === nothing ? ones(n) : Float64.(collect(D)))
    fn = _base_fn(base)
    fac = V === nothing ? nothing :
          _topk_factor_nodes(V, n, qa, "abilities_from_topk")
    logit_t = log.(target) .- log1p.(-target)
    logt = log.(target)
    mu = -(logt .- sum(logt) / n) ./ 2
    alpha = n > 2 ? 1.0 : 0.7
    resid_max = Inf
    iters = 0
    for it in 1:n_iter
        iters = it
        local qraw, sl
        if fac === nothing
            ws = _topk_with_slopes(mu, sd, k, fn, points)
            qraw, sl = ws.q, ws.slopes
        else
            qraw = zeros(n)
            sl = zeros(n)
            for j in 1:size(fac.nodes, 1)
                ws = _topk_with_slopes(mu .+ fac.Vm * fac.nodes[j, :],
                                       sd, k, fn, points)
                qraw .+= fac.w[j] .* ws.q
                sl .+= fac.w[j] .* ws.slopes
            end
        end
        qhat = _checked_topk(qraw, k, "top-k inversion")
        resid = (log.(max.(qhat, TINY)) .- log.(max.(1 .- qhat, TINY))) .- logit_t
        resid_max = maximum(abs.(resid))
        resid_max < tol && break
        dlogit = min.(sl ./ max.(qhat .* (1 .- qhat), TINY), -1e-6)
        lim = min.(2, 10 .* abs.(resid))
        mu .-= clamp.(alpha .* resid ./ dlogit, -lim, lim)
        mu .-= sum(mu) / n
    end
    converged = resid_max < tol
    return_info && return (mu = mu, converged = converged,
                           max_logit_residual = resid_max,
                           iterations = iters, floored = vt.floored)
    converged || @warn "abilities_from_topk did not converge" resid_max iters
    return mu
end

"""Joint (loc, scale) from two membership curves; refuses fixed
loadings (two curves are then one equation short)."""
function loc_scale_from_topk_pair(q1, k1, q2, k2; D0 = nothing,
                                  base = "normal", points = 513,
                                  n_iter = 60, tol = 1e-8, ridge = 0.0,
                                  mu0 = nothing, V = nothing,
                                  return_info = false)
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
    sd = D0 === nothing ? ones(n) : sqrt.(Float64.(collect(D0)))
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
    converged = resid_max < tol || (sqr > 0 && !last_accepted)
    return_info && return (mu = mu, sd = sd, converged = converged,
                           max_logit_residual = resid_max,
                           iterations = iters)
    converged || @warn "loc_scale_from_topk_pair did not converge" resid_max
    return (mu = mu, sd = sd)
end

_I(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

"""Win plus EXACTLY-second marginals: P(2nd) + P(win) = P(top-2)."""
function loc_scale_from_win_and_second(p_win, p_second; kwargs...)
    n = length(p_win)
    length(p_second) == n || error("p_win and p_second must have equal length")
    (any(p_win .<= 0) || any(p_second .<= 0)) &&
        error("all win and second probabilities must be positive")
    p1 = p_win ./ sum(p_win)
    top2 = p1 .+ p_second ./ sum(p_second)
    return loc_scale_from_topk_pair(p1, 1, top2, 2; kwargs...)
end

function _rank_marginal_with_jacobian(mu, sd, r, fn, points)
    n = length(mu)
    g = _topk_grid(mu, sd, n - 1, fn, points)
    dens = g.f' ./ sd                            # (n, L)
    C = _count_distribution(g.F)
    p = zeros(n)
    J = zeros(n, n)
    for i in 1:n
        Qi = _loo_pmf(C, g.F, i)
        p[i] = sum(Qi[:, r] .* dens[i, :]) * g.dx
        # P(N_{-ij} = n-1) is identically zero (only n-2 others exist)
        pair = r <= n - 1 ? _pair_pmf_at(Qi, g.F, i, r) :
               zeros(n, length(g.x))
        if r >= 2
            pair = pair .- _pair_pmf_at(Qi, g.F, i, r - 1)
        end
        kern = pair .* dens[i, :]'
        row = vec(sum(kern .* dens, dims = 2)) .* g.dx
        row[i] = 0.0
        row[i] = -sum(row)
        J[i, :] .= row
    end
    return (p = p, J = J)
end

"""Invert one EXACT-rank marginal at frozen scales (two-branched for
r >= 2; mu0 selects the branch)."""
function abilities_from_rank_marginal(p, r; mu0 = nothing, D = nothing,
                                      base = "normal", points = 513,
                                      n_iter = 60, tol = 1e-8,
                                      return_info = false)
    n = length(p)
    r = Int(r)
    1 <= r <= n || error("rank must be in [1, n]; got r=$r, n=$n")
    any(p .<= 0) && error("all rank probabilities must be positive")
    logt = log.(p ./ sum(p))
    sd = sqrt.(D === nothing ? ones(n) : Float64.(collect(D)))
    fn = _base_fn(base)
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
    return_info && return (mu = mu, converged = converged,
                           max_log_residual = resid_max, iterations = iters)
    converged || @warn "abilities_from_rank_marginal did not converge" resid_max
    return mu
end
