# Emit one ACCEPT/REFUSE verdict per case, for the julia port.
root = dirname(dirname(abspath(ARGS[1])))
include(joinpath(root, "julia", "winning", "src", "winning.jl"))
using .winning
include(joinpath(root, "julia", "winning", "test", "minijson.jl"))
d = parse_json(read(ARGS[1], String))
num(x) = x isa AbstractVector ? Float64[num(v) for v in x] :
         (x isa AbstractString ? parse(Float64, x) : Float64(x))
parts = String[]
for cs in d["cases"]
    id = cs["id"]
    verdict, val = try
        mu = haskey(cs, "mu") ? num(cs["mu"]) : nothing
        D  = haskey(cs, "D")  ? num(cs["D"])  : nothing
        V  = haskey(cs, "Vt") ? permutedims(reduce(hcat, [num(r) for r in cs["Vt"]])) :
             haskey(cs, "V") ? (cs["V"] isa AbstractVector && cs["V"][1] isa AbstractVector ?
                 permutedims(reduce(hcat, [num(r) for r in cs["V"]])) : num(cs["V"])) : nothing
        p = if cs["verb"] == "hermite"
                hermite_nodes(Int(num(cs["r"])), Int(num(cs["order"]))).F
            elseif cs["verb"] == "rank"
                rank_probabilities(mu; D = D)
            elseif cs["verb"] == "bottomk"
                bottom_k_probabilities(mu, Int(num(cs["k"])); D = D)
            elseif cs["verb"] == "race"
                race_probabilities(mu; V = V, D = D)
            elseif cs["verb"] == "inverse"
                abilities_from_race(num(cs["p"]); D = D)
            else
                top_k_probabilities(mu, Int(num(cs["k"])); D = D)
            end
        v = cs["verb"] == "hermite" ?
            Float64[size(p, 1), size(p, 2)] :
            Float64[x for x in (ndims(p) == 2 ? vec(permutedims(p)) :
                                vec(collect(p)))[1:min(6, length(p))]]
        (all(isfinite, p) ? "ACCEPT" : "ACCEPT_NONFINITE",
         all(isfinite, v) ? v : Float64[])
    catch e
        ("REFUSE", Float64[])
    end
    # values, so a case both ports ACCEPT and answer DIFFERENTLY is
    # caught too: agreeing to accept is not agreeing on the race
    vs = join([string(round(x, digits = 12)) for x in val], ",")
    push!(parts, "\"$id\":{\"verdict\":\"$verdict\",\"value\":[$vs]}")
end
println("{" * join(parts, ",") * "}")
