# Julia parity checker: rebuilds the same-named scenarios as
# gen_vectors.py from the embedded inputs and asserts agreement with
# the python reference. Scenarios outside the Julia port's scope so
# far (blocks/nested/tree, polish, classic, coph) are reported as
# skipped. Dependency-free, including the JSON reader below.
#
#   julia --project=julia/winning parity/check.jl

include(joinpath(@__DIR__, "..", "julia", "winning", "src", "winning.jl"))
using .winning
include(joinpath(@__DIR__, "..", "julia", "winning", "test", "minijson.jl"))
include(joinpath(@__DIR__, "..", "julia", "winning", "test", "parity.jl"))

fails, skipped = run_parity(joinpath(@__DIR__, "vectors.json"))
fails > 0 && exit(1)
