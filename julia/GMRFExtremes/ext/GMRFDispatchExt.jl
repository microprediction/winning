# Loading GaussianMarkovRandomFields.jl arms dispatch on its GMRF
# types: any AbstractGMRF whose precision_matrix is tridiagonal
# converts to a GaussMarkovChain (the refusal contract gates the
# rest), and the query verbs accept the GMRF directly.
module GMRFDispatchExt

using GMRFExtremes
using GaussianMarkovRandomFields: AbstractGMRF, mean, precision_matrix

GMRFExtremes.GaussMarkovChain(g::AbstractGMRF) =    chain_from_precision(mean(g), precision_matrix(g))

for f in (:max_cdf, :excursion_probability, :first_passage)
    @eval GMRFExtremes.$f(g::AbstractGMRF, u; kw...) =
        GMRFExtremes.$f(GaussMarkovChain(g), u; kw...)
end
GMRFExtremes.argmax_marginals(g::AbstractGMRF; kw...) =
    argmax_marginals(GaussMarkovChain(g); kw...)
GMRFExtremes.expected_max(g::AbstractGMRF; kw...) =
    expected_max(GaussMarkovChain(g); kw...)

end # module
