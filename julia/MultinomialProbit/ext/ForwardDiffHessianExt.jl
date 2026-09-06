# Loading ForwardDiff upgrades loglik_hessian from central differences
# of the analytic score (~1e-8) to the exact dual-mode derivative of
# that same score (forward-over-analytic, machine precision).
module ForwardDiffHessianExt

using MultinomialProbit
using ForwardDiff

function __init__()
    MultinomialProbit.HESSIAN_ENGINE[] = (m, theta) -> begin
        g(t) = -MultinomialProbit._nll_grad(m, t)[2]
        Hs = ForwardDiff.jacobian(g, theta)
        (Hs .+ Hs') ./ 2
    end
end

end # module
