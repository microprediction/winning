# Loading MvNormalCDF arms the dense fallback: refused cases (dense
# covariance, rank > 2, sharpness > 3) route to the genuine incumbent
# instead of erroring. The hook returns (p, e) in the incumbent's
# convention.
module MvNormalCDFFallbackExt

using MvNormalCDFFast
using MvNormalCDF

function __init__()
    MvNormalCDFFast.DENSE_FALLBACK[] = (lower, upper, mean, sigma) -> begin
        n = size(sigma, 1)
        mu = mean === nothing ? zeros(n) : Float64.(collect(mean))
        lo = lower === nothing ? fill(-Inf, n) : Float64.(collect(lower))
        up = upper === nothing ? fill(Inf, n) : Float64.(collect(upper))
        MvNormalCDF.mvnormcdf(mu, Float64.(Matrix(sigma)), lo, up)
    end
end

end # module
