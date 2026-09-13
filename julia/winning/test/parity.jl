# The parity scenarios against the Python reference, shared by
# Pkg.test (runtests.jl) and the standalone parity/check.jl. Expects
# parse_json from minijson.jl and the winning exports in scope.

vec1(x) = Float64.(x)
mat(x) = permutedims(hcat([Float64.(r) for r in x]...))
flat(x) = x isa AbstractMatrix ? vec(permutedims(x)) :
          x isa AbstractArray ? vcat((flat(e) for e in x)...) : [Float64(x)]

"""
    run_parity(vectors_path) -> (fails, skipped)

Rebuild every scenario of parity/gen_vectors.py from the embedded
inputs and compare with the Python value at the recorded tolerance.
Scenarios outside the Julia port's scope so far (blocks/nested/tree,
polish, classic, coph) are reported as skipped.
"""
function run_parity(vectors_path::AbstractString)
    vecs = parse_json(read(vectors_path, String))
    inp = vecs["inputs"]
    sc = vecs["scenarios"]
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
            fails += 1
            continue
        end
        d = maximum(abs.(got .- ref))
        ok = d <= tol
        println("$(ok ? "ok  " : "FAIL")  $(rpad(name, 28)) max|diff| " *
                "$(round(d, sigdigits = 3))  (tol $tol)")
        ok || (fails += 1)
    end
    println("skipped (not yet ported): ", join(skipped, ", "))
    if fails > 0
        println("$fails parity failures")
    else
        println("all ported scenarios match the python reference")
    end
    return fails, skipped
end
