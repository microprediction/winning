# winning (R package)

All N win probabilities of a factor-structured Gaussian race
(multinomial probit with covariance VV' + diag(D)) in one shared-lattice
pass, and the inverse: calibrating abilities from observed shares.

```r
nodes <- hermite_nodes(2)
p <- win_probabilities_factor(mu, V, D, nodes)      # min wins; negate mu for argmax
a <- abilities_from_probabilities_factor(p, V, D, nodes)
```

Why not N calls to mvtnorm::pmvnorm? That computes the same numbers
(agreement to ~1e-6 at N = 5, verified in the test suite) but pays for N
separate (N-1)-dimensional integrals; the shared survival field prices
every alternative from one O(Q N L) pass. At N = 200 the difference is
seconds versus minutes; at N = 1000, seconds versus hours.

Pure base R, no dependencies (mvtnorm suggested for the cross-check
tests only). The Python/Rust implementations, benchmarks, and the paper
live in the parent repository.

CRAN preparation checklist: roxygenize man pages, add vignette,
R CMD check --as-cran, submit.
