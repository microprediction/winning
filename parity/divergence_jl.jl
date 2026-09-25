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
        V  = haskey(cs, "V")  ? num(cs["V"])  : nothing
        p = if cs["verb"] == "race"
                race_probabilities(mu; V = V, D = D)
            elseif cs["verb"] == "inverse"
                abilities_from_race(num(cs["p"]); D = D)
            else
                top_k_probabilities(mu, Int(round(num(cs["k"]))); D = D)
            end
        (all(isfinite, p) ? "ACCEPT" : "ACCEPT_NONFINITE", collect(p)[1:min(6,end)])
    catch e
        ("REFUSE", Float64[])
    end
    push!(parts, "\"$id\":{\"verdict\":\"$verdict\"}")
end
println("{" * join(parts, ",") * "}")
