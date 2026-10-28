# Loading GaussianMarkovRandomFields.jl arms dispatch on its GMRF
# types: any AbstractGMRF whose precision_matrix is tridiagonal
# converts to a GaussMarkovChain (the refusal contract gates the
# rest), and the query verbs accept the GMRF directly.
module GMRFDispatchExt

using GMRFExtremes
using GaussianMarkovRandomFields: AbstractGMRF, mean, precision_matrix,
    ConstrainedGMRF, WorkspaceGMRF, ConstraintInfo

GMRFExtremes.GaussMarkovChain(g::AbstractGMRF) =    chain_from_precision(mean(g), precision_matrix(g))

# A hard constraint Ax = e is not in precision_matrix(g): for these types
# it returns the unconstrained base precision while mean(g) is the
# constrained mean, so the generic conversion above would price a
# different law (#804). Refuse rather than drop the constraint.
const _HardConstrained = Union{ConstrainedGMRF,
    WorkspaceGMRF{<:Any, <:Any, <:Any, <:ConstraintInfo}}
GMRFExtremes.GaussMarkovChain(g::_HardConstrained) = throw(ArgumentError(
    "GMRFExtremes does not support hard linear constraints (got a " *
    "$(nameof(typeof(g))) with Ax = e): its precision_matrix is the " *
    "unconstrained base precision, so the extrema would be those of a " *
    "different law. Use the unconstrained GMRF, or condition on the " *
    "constraint first and pass the resulting chain or precision."))

for f in (:max_cdf, :excursion_probability, :first_passage)
    @eval GMRFExtremes.$f(g::AbstractGMRF, u; kw...) =
        GMRFExtremes.$f(GaussMarkovChain(g), u; kw...)
end
GMRFExtremes.argmax_marginals(g::AbstractGMRF; kw...) =
    argmax_marginals(GaussMarkovChain(g); kw...)
GMRFExtremes.expected_max(g::AbstractGMRF; kw...) =
    expected_max(GaussMarkovChain(g); kw...)

end # module
