# Julia parity checker: rebuilds the same-named scenarios as
# gen_vectors.py from the embedded inputs and asserts agreement with
# the python reference. Scenarios outside the Julia port's scope so
# far (blocks/nested/tree, polish, classic, coph) are reported as
# skipped. Dependency-free, including the JSON reader below.
#
#   julia --project=julia/winning parity/check.jl

include(joinpath(@__DIR__, "..", "julia", "winning", "src", "winning.jl"))
using .winning

# ---- minimal JSON reader (objects/arrays/numbers/strings/consts) ----
mutable struct P
    s::String
    i::Int
end
peek(p) = p.s[p.i]
function skipws(p)
    while p.i <= lastindex(p.s) && isspace(p.s[p.i])
        p.i += 1
    end
end
function jval(p)
    skipws(p)
    c = peek(p)
    c == '{' && return jobj(p)
    c == '[' && return jarr(p)
    c == '"' && return jstr(p)
    if startswith(SubString(p.s, p.i), "true")
        p.i += 4; return true
    elseif startswith(SubString(p.s, p.i), "false")
        p.i += 5; return false
    elseif startswith(SubString(p.s, p.i), "null")
        p.i += 4; return nothing
    end
    j = p.i
    while j <= lastindex(p.s) && (p.s[j] in "+-.eE0123456789")
        j += 1
    end
    v = parse(Float64, p.s[p.i:(j - 1)])
    p.i = j
    return v
end
function jstr(p)
    p.i += 1
    out = IOBuffer()
    while peek(p) != '"'
        c = p.s[p.i]
        if c == '\\'
            p.i += 1
            c = p.s[p.i]
        end
        write(out, c)
        p.i += 1
    end
    p.i += 1
    return String(take!(out))
end
function jarr(p)
    p.i += 1
    out = Any[]
    skipws(p)
    if peek(p) == ']'
        p.i += 1
        return out
    end
    while true
        push!(out, jval(p))
        skipws(p)
        if peek(p) == ','
            p.i += 1
        else
            p.i += 1
            return out
        end
    end
end
function jobj(p)
    p.i += 1
    out = Dict{String,Any}()
    skipws(p)
    if peek(p) == '}'
        p.i += 1
        return out
    end
    while true
        skipws(p)
        k = jstr(p)
        skipws(p)
        p.i += 1                       # ':'
        out[k] = jval(p)
        skipws(p)
        if peek(p) == ','
            p.i += 1
        else
            p.i += 1
            return out
        end
    end
end

vec1(x) = Float64.(x)
mat(x) = permutedims(hcat([Float64.(r) for r in x]...))
flat(x) = x isa AbstractMatrix ? vec(permutedims(x)) :
          x isa AbstractArray ? vcat((flat(e) for e in x)...) : [Float64(x)]

const vecs = jval(P(read(joinpath(@__DIR__, "vectors.json"), String), 1))
const inp = vecs["inputs"]
const sc = vecs["scenarios"]

mu = vec1(inp["mu"])
V1 = mat(inp["V1"])
V2 = mat(inp["V2"])
D = vec1(inp["D"])
pt = vec1(inp["p_target"])
sdt = vec1(inp["sd_true"])
gumD = fill(pi^2 / 6, length(mu))
R = mat(sc["rank_marginals"]["value"])
q2v = vec1(sc["topk2_normal"]["value"])
lw = vec1(sc["loc_scale_win"]["value"])
lp = vec1(sc["loc_scale_place"]["value"])
it2 = vec1(sc["invert_topk2"]["value"])

runs = Dict{String,Function}(
    "independent_normal" => () -> race_probabilities(mu; D = D, points = 257),
    "factor1_normal" => () -> race_probabilities(mu; V = V1, D = D, points = 257),
    "factor2_normal" => () -> race_probabilities(mu; V = V2, D = D, points = 257),
    "factor2_slopes" => () -> race_probabilities(mu; V = V2, D = D,
        points = 257, return_slopes = true).slopes,
    "factor2_span" => () -> race_probabilities(mu; V = V2, D = D,
        points = 501, window = "span"),
    "gumbel_independent" => () -> race_probabilities(mu; D = gumD,
        base = "gumbel", points = 1001),
    "invert_factor" => () -> abilities_from_race(pt; V = V1, D = D,
        points = 257),
    "topk2_normal" => () -> top_k_probabilities(mu, 2; D = D, points = 257),
    "topk4_gumbel" => () -> top_k_probabilities(mu, 4; D = gumD,
        base = "gumbel", points = 1001),
    "topk2_jacobian_mu" => () -> top_k_jacobians(mu, 2; D = D,
        points = 257).Jmu,
    "topk2_jacobian_sigma" => () -> top_k_jacobians(mu, 2; D = D,
        points = 257).Jsigma,
    "topk2_jacobian_mu_factor" => () -> top_k_jacobians(mu, 2; D = D,
        V = V1, points = 257).Jmu,
    "topk2_jacobian_sigma_factor" => () -> top_k_jacobians(mu, 2; D = D,
        V = V1, points = 257).Jsigma,
    "invert_topk2" => () -> abilities_from_topk(q2v, 2; D = D, points = 257),
    "loc_scale_win" => () -> top_k_probabilities(mu, 1; D = sdt .^ 2,
        points = 257),
    "loc_scale_place" => () -> top_k_probabilities(mu, 3; D = sdt .^ 2,
        points = 257),
    "loc_scale_mu" => () -> loc_scale_from_topk_pair(lw, 1, lp, 3;
        points = 257).mu,
    "loc_scale_sd" => () -> loc_scale_from_topk_pair(lw, 1, lp, 3;
        points = 257).sd,
    "loc_scale_ridge_mu" => () -> loc_scale_from_topk_pair(lw, 1, lp, 3;
        ridge = 0.05, points = 257).mu,
    "loc_scale_ridge_sd" => () -> loc_scale_from_topk_pair(lw, 1, lp, 3;
        ridge = 0.05, points = 257).sd,
    "rank_marginals" => () -> rank_probabilities(mu; D = D, points = 257),
    "win_second_mu" => () -> loc_scale_from_win_and_second(R[:, 1], R[:, 2];
        points = 257).mu,
    "win_second_sd" => () -> loc_scale_from_win_and_second(R[:, 1], R[:, 2];
        points = 257).sd,
    "invert_second" => () -> abilities_from_rank_marginal(R[:, 2], 2;
        mu0 = it2, D = D, points = 257),
)

fails = 0
skipped = String[]
for name in sort(collect(keys(sc)))
    if !haskey(runs, name)
        push!(skipped, name)
        continue
    end
    ref = flat(sc[name]["value"])
    tol = Float64(sc[name]["tol"])
    got = try
        flat(runs[name]())
    catch e
        println("FAIL  $(rpad(name, 28)) error: $e")
        global fails += 1
        continue
    end
    d = maximum(abs.(got .- ref))
    ok = d <= tol
    println("$(ok ? "ok  " : "FAIL")  $(rpad(name, 28)) max|diff| " *
            "$(round(d, sigdigits = 3))  (tol $tol)")
    ok || (global fails += 1)
end
println("skipped (not yet ported): ", join(skipped, ", "))
if fails > 0
    println("$fails parity failures")
    exit(1)
end
println("all ported scenarios match the python reference")
