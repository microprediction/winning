# The measured rows of the checks table in an EARLIER version of the
# win_nodes paper. The Gauss-Markov chain material was cut from that
# paper on 2026-11-09 and papers/win_nodes/paper.tex no longer cites
# these rows; they are kept as the record for the GMRFExtremes.jl pass
# (not the exp1-exp3 Python prototypes). Seeded Monte Carlo referees;
# the i.i.d. row has a closed form.
#   julia run_checks.jl > results.txt
include(joinpath(@__DIR__, "..", "..", "..", "julia", "GMRFExtremes",
                 "src", "GMRFExtremes.jl"))
using .GMRFExtremes
const GE = GMRFExtremes
using Random: Xoshiro
using Printf

ar(phi, n) = GaussMarkovChain(zeros(n), 1.0, fill(phi, n - 1),
                              fill(sqrt(1 - phi^2), n - 1))

function simulate(c, m, rng)
    n = length(c)
    X = zeros(m, n)
    X[:, 1] .= c.mu[1] .+ c.sd0 .* randn(rng, m)
    for t in 1:(n - 1)
        X[:, t + 1] .= c.mu[t + 1] .+ c.phi[t] .* (X[:, t] .- c.mu[t]) .+
                       c.s[t] .* randn(rng, m)
    end
    return X
end

function mc_argmax(X)
    n = size(X, 2)
    counts = zeros(n)
    for r in 1:size(X, 1)
        counts[argmax(@view X[r, :])] += 1
    end
    return counts ./ size(X, 1)
end

M = 400_000

# 1. stationary AR(1), phi = 0.9, n = 20: the boundary pile-up
c = ar(0.9, 20)
P = argmax_marginals(c; points = 400)
ref = mc_argmax(simulate(c, M, Xoshiro(1)))
@printf("argmax AR(1) phi=0.9 n=20 (L=400): end %.4f  mid %.4f  ratio %.3f\n",
        P[1], P[10], (P[1] + P[end]) / 2 / P[10])
@printf("  MC %d: end %.4f  mid %.4f   max|exact-MC| = %.2e\n",
        M, ref[1], ref[10], maximum(abs.(P .- ref)))

# 2. random walk, n = 40
rw = GaussMarkovChain(zeros(40), 1.0, ones(39), ones(39))
Prw = argmax_marginals(rw; points = 400)
refrw = mc_argmax(simulate(rw, M, Xoshiro(2)))
@printf("argmax random walk n=40 (L=400): max|exact-MC| = %.2e\n",
        maximum(abs.(Prw .- refrw)))

# 3. i.i.d. closed form, n = 12
iid = ar(0.0, 12)
errs = [abs(max_cdf(iid, u) - GE.ndtr(u)^12) for u in -0.5:0.25:2.5]
@printf("max-CDF i.i.d. n=12 (L=400) vs Phi(u)^12: max err %.1e\n",
        maximum(errs))

# 4. the inhomogeneous ten-node chain of the package's MC referee test
mu = [0.0, 0.3, 0.6, 0.4, 0.0, -0.3, -0.1, 0.2, 0.5, 0.1]
c = GaussMarkovChain(mu, 0.8, fill(0.7, 9), vcat(fill(0.6, 4), fill(0.9, 5)))
X = simulate(c, M, Xoshiro(1))
mx = vec(maximum(X, dims = 2))
e = [abs(max_cdf(c, u) - sum(mx .<= u) / M) for u in (0.5, 1.5, 2.5)]
@printf("inhomogeneous n=10, MC %d draws\n", M)
@printf("  max-CDF at u = 0.5, 1.5, 2.5: errs %.1e %.1e %.1e\n", e...)
m = sum(mx) / M
se = sqrt(sum((mx .- m) .^ 2) / M) / sqrt(M)
em = expected_max(c)
@printf("  E[max] trapezoid %.4f  MC %.4f  err %.1e  (MC s.e. %.1e)\n",
        em, m, abs(em - m), se)
# the left-rectangle rule on the same threshold grid as expected_max
msd = GE.marginal_sd(c)
nu = 200
lo = minimum(mu) - 8 * maximum(msd)
hi = maximum(mu) + 8 * maximum(msd)
us = collect(range(lo, hi, length = nu))
du = us[2] - us[1]
F = max_cdf(c, us)
rect = lo + sum(1 .- F) * du
@printf("  E[max] left rectangle %.4f  bias vs MC %+.3f  (du = %.3f)\n",
        rect, rect - m, du)
fp = first_passage(c, 1.0)
hit = [findfirst(>(1.0), @view X[r, :]) for r in 1:M]
fpe = maximum(abs(fp[t] - sum(h -> h !== nothing && h == t, hit) / M)
              for t in 1:10)
@printf("  first passage u=1.0, all 10 indices: max err %.1e\n", fpe)
