# Run by R CMD check through tests/testthat.R, so the package is
# already attached and its internals are in scope. These files used
# to source(file.path("..", "..", "R", ...)) instead, which only
# works from the repo and fails inside a check -- which is how they
# went unrun (#142).
test_that("factor case matches mvtnorm within its own error bound", {
  set.seed(11)
  n <- 30
  V <- matrix(rnorm(n * 2), n, 2) * 0.6
  D <- 0.5 + runif(n)
  b <- rnorm(n, 1)
  pf <- pmvnorm_fast(upper = b, V = V, D = D)
  pm <- mvtnorm::pmvnorm(upper = b, sigma = V %*% t(V) + diag(D),
                         algorithm = mvtnorm::GenzBretz(maxpts = 5e5,
                                                        abseps = 1e-9))
  expect_equal(attr(pf, "method"), "factor")
  expect_lt(abs(as.numeric(pf) - as.numeric(pm)), 1e-7)
})

test_that("two-sided rectangles agree", {
  set.seed(12)
  n <- 10
  V <- matrix(rnorm(n * 2), n, 2) * 0.6
  D <- 0.5 + runif(n)
  a <- rnorm(n, -1.5); b <- a + abs(rnorm(n)) + 0.5
  pf <- pmvnorm_fast(a, b, V = V, D = D)
  pm <- mvtnorm::pmvnorm(lower = a, upper = b,
                         sigma = V %*% t(V) + diag(D),
                         algorithm = mvtnorm::GenzBretz(maxpts = 5e5,
                                                        abseps = 1e-9))
  expect_lt(abs(as.numeric(pf) - as.numeric(pm)), 1e-7)
})

test_that("exactly-structured sigma is detected; dense falls back", {
  set.seed(13)
  n <- 8
  V <- matrix(rnorm(n), n, 1)
  D <- 0.3 + runif(n)
  p1 <- pmvnorm_fast(upper = rnorm(n, 1), sigma = V %*% t(V) + diag(D))
  expect_equal(attr(p1, "method"), "factor")
  Sd <- crossprod(matrix(rnorm(n * n), n, n)) + diag(n)
  p2 <- pmvnorm_fast(upper = rnorm(n, 1), sigma = Sd)
  expect_equal(attr(p2, "method"), "mvtnorm-fallback")
})

test_that("deep tails recenter and agree with minimax tilting", {
  skip_if_not_installed("TruncatedNormal")
  set.seed(7)
  n <- 200
  V <- matrix(rnorm(n * 2), n, 2) * 0.4
  D <- 0.5 + runif(n)
  b <- rnorm(n, 1.5)
  pf <- pmvnorm_fast(upper = b, V = V, D = D)
  pb <- TruncatedNormal::pmvnorm(mu = rep(0, n),
                                 sigma = V %*% t(V) + diag(D),
                                 lb = rep(-Inf, n), ub = b, B = 5e4)
  expect_equal(attr(pf, "method"), "factor-recentered")
  expect_lt(abs(as.numeric(pf) - as.numeric(pb)) / as.numeric(pb), 2e-2)
})

test_that("independence sanity: product of marginals", {
  n <- 6
  D <- 0.5 + (1:n) / 10
  b <- seq(-1, 1.5, length.out = n)
  pf <- pmvnorm_fast(upper = b, V = matrix(0, n, 1), D = D)
  expect_lt(abs(as.numeric(pf) - prod(pnorm(b / sqrt(D)))), 1e-12)
})

# --- empty and degenerate rectangles (#235) ---------------------------
#
# P(lower <= X <= upper) over a reversed coordinate is an EMPTY event.
# This formed the negative conditional cell pnorm(0) - pnorm(1) and
# clamped it to the underflow floor, so an impossible observation came
# back as 1e-300 -- a finite log probability near -690.8 rather than
# -Inf. mvtnorm::pmvnorm, which this package is a drop-in for, raises on
# reversed bounds and returns 0 when a coordinate has lower == upper.

test_that("reversed bounds raise rather than returning the floor", {
  expect_error(
    pmvnorm_fast(lower = 1, upper = 0, mean = 0,
                 V = matrix(0, 1, 1), D = 1),
    "lower must not exceed upper")
})

test_that("the error names the offending coordinate", {
  expect_error(
    pmvnorm_fast(lower = c(0, 1, -1), upper = c(1, 0, 1),
                 mean = rep(0, 3), V = matrix(0, 3, 1), D = rep(1, 3)),
    "coordinate 2")
})

test_that("a degenerate coordinate has exactly zero mass", {
  p <- pmvnorm_fast(lower = 0.5, upper = 0.5, mean = 0,
                    V = matrix(0, 1, 1), D = 1)
  expect_identical(as.numeric(p), 0)
  expect_equal(attr(p, "method"), "degenerate-rectangle")
  expect_true(is.infinite(log(as.numeric(p))))
})

test_that("one degenerate coordinate among valid ones is still zero", {
  p <- pmvnorm_fast(lower = c(0, 0.3, -1), upper = c(1, 0.3, 1),
                    mean = rep(0, 3), V = matrix(0, 3, 1), D = rep(1, 3))
  expect_identical(as.numeric(p), 0)
})

test_that("valid rectangles are untouched by the check", {
  n <- 4
  D <- 0.6 + (1:n) / 10
  lo <- rep(-1, n); up <- seq(0.2, 1.4, length.out = n)
  p <- pmvnorm_fast(lower = lo, upper = up, V = matrix(0, n, 1), D = D)
  expect_lt(abs(as.numeric(p) -
                prod(pnorm(up / sqrt(D)) - pnorm(lo / sqrt(D)))), 1e-12)
})

# --- deterministic coordinates (#206) ---------------------------------
#
# A coordinate with zero idiosyncratic variance is DETERMINISTIC given
# the factor draw, so its conditional cell is an indicator, not a normal
# interval. Dividing by s = 0 is 0/0 = NaN the moment the deterministic
# value lands exactly on an inclusive boundary: the documented
# P(X_1 <= 0) = 1 for X_1 identically 0.

test_that("a deterministic coordinate on the boundary is included", {
  p <- pmvnorm_fast(lower = c(-Inf, -Inf), upper = c(0, 0),
                    mean = c(0, 0), V = matrix(0, 2, 1), D = c(0, 1))
  expect_false(is.na(as.numeric(p)))
  expect_lt(abs(as.numeric(p) - 0.5), 1e-12)
})

test_that("a deterministic coordinate strictly inside is included", {
  p <- pmvnorm_fast(lower = c(-Inf, -Inf), upper = c(1, 0),
                    mean = c(0, 0), V = matrix(0, 2, 1), D = c(0, 1))
  expect_lt(abs(as.numeric(p) - 0.5), 1e-12)
})

test_that("a deterministic coordinate outside gives exactly zero", {
  p <- pmvnorm_fast(lower = c(-Inf, -Inf), upper = c(-1, 0),
                    mean = c(0, 0), V = matrix(0, 2, 1), D = c(0, 1))
  expect_identical(as.numeric(p), 0)
  expect_equal(attr(p, "method"), "outside-support")
  q <- pmvnorm_fast(lower = c(0.5, -Inf), upper = c(Inf, 0),
                    mean = c(0, 0), V = matrix(0, 2, 1), D = c(0, 1))
  expect_identical(as.numeric(q), 0)
})

test_that("every coordinate deterministic and inside is one", {
  p <- pmvnorm_fast(lower = c(-Inf, -Inf), upper = c(0, 0),
                    mean = c(0, 0), V = matrix(0, 2, 1), D = c(0, 0))
  expect_lt(abs(as.numeric(p) - 1), 1e-12)
})

test_that("a deterministic coordinate that still loads on the factor", {
  # D_1 = 0 but V_1 != 0: constant only GIVEN f, so the indicator
  # restricts the factor region and the answer comes out of quadrature.
  # P = E_f[ 1{f <= 0} Phi(-f/2) ], by one-dimensional integration.
  p <- pmvnorm_fast(lower = c(-Inf, -Inf), upper = c(0, 0), mean = c(0, 0),
                    V = matrix(c(0.5, 0.5), 2, 1), D = c(0, 1))
  exact <- integrate(function(t) dnorm(t) * pnorm(-0.5 * t),
                     -50, 0)$value
  expect_lt(abs(as.numeric(p) - exact), 1e-4)
})

# --- per-coordinate arguments are not recycled (#285) -----------------
#
# R recycles a short vector silently whenever its length divides n, and
# emits no warning, so `D = c(1, 4)` at n = 4 became c(1,4,1,4) and the
# call returned [Phi(1)Phi(1/2)]^2 = 0.3384 -- a perfectly plausible
# probability for a Gaussian the caller never described. `mean` and
# `upper` did the same. The python reference validates through
# as_idio/as_loadings and julia fails on unequal lengths; only this port
# guessed. A scalar is still broadcast on purpose: that is what the
# -Inf/Inf defaults are.

test_that("a wrong-length D is refused rather than recycled", {
  V <- matrix(0, 4, 1)
  for (bad in list(c(1, 4), rep(1, 2), rep(1, 5), rep(1, 3))) {
    expect_error(pmvnorm_fast(upper = rep(1, 4), mean = rep(0, 4),
                              V = V, D = bad),
                 "D must be a scalar or one value per coordinate")
  }
  # and the message says what it got and what it needed
  expect_error(pmvnorm_fast(upper = rep(1, 4), V = V, D = c(1, 4)),
               "got 2 for n = 4")
})

test_that("a wrong-length mean is refused rather than recycled", {
  V <- matrix(0, 4, 1)
  expect_error(pmvnorm_fast(upper = rep(0, 4), mean = c(0, 1),
                            V = V, D = rep(1, 4)),
               "mean must be a scalar or one value per coordinate")
})

test_that("wrong-length bounds are refused too", {
  # found by sweeping the pattern rather than the reported symptom:
  # lower/upper went through rep_len, which recycles in exactly the
  # same silence
  V <- matrix(0, 4, 1)
  expect_error(pmvnorm_fast(upper = c(0, 1), V = V, D = rep(1, 4)),
               "upper must be a scalar or one value per coordinate")
  expect_error(pmvnorm_fast(lower = c(-1, 0), upper = rep(1, 4),
                            V = V, D = rep(1, 4)),
               "lower must be a scalar or one value per coordinate")
})

test_that("a negative or missing variance is refused by name", {
  V <- matrix(0, 4, 1)
  expect_error(pmvnorm_fast(upper = rep(1, 4), V = V, D = c(1, 1, -1, 1)),
               "D\\[3\\] = -1 is a negative variance")
  expect_error(pmvnorm_fast(upper = rep(1, 4), V = V,
                            D = c(1, NA_real_, 1, 1)),
               "D has a missing entry")
  expect_error(pmvnorm_fast(upper = rep(1, 4), V = V, D = c(1, Inf, 1, 1)),
               "D has a non-finite entry")
})

test_that("the documented spellings all still work and agree", {
  V <- matrix(0, 4, 1)
  want <- prod(pnorm(rep(1, 4)))
  # scalar D, length-4 D, and the default bounds are one answer
  a <- pmvnorm_fast(upper = rep(1, 4), V = V, D = 1)
  b <- pmvnorm_fast(upper = rep(1, 4), V = V, D = rep(1, 4))
  cc <- pmvnorm_fast(upper = rep(1, 4), mean = rep(0, 4), V = V,
                     D = rep(1, 4))
  expect_lt(abs(as.numeric(a) - want), 1e-12)
  expect_identical(as.numeric(a), as.numeric(b))
  expect_identical(as.numeric(a), as.numeric(cc))
  # an infinite bound is legal; an infinite MEAN is not
  expect_true(is.finite(as.numeric(
    pmvnorm_fast(lower = -Inf, upper = rep(1, 4), V = V, D = 1))))
  expect_error(pmvnorm_fast(upper = rep(1, 4), mean = c(0, 0, Inf, 0),
                            V = V, D = 1),
               "mean has a non-finite entry")
})

# --- the rectangle depends on the Gaussian, not its spelling ----------
#
# Same fixtures as tests/test_fastmvn_semantics.py in the python tree.

test_that("where the variance is stored does not matter (#429)", {
  for (b in c(-3, -4, -6)) {
    exact <- pnorm(b)^2
    in_v <- pmvnorm_fast(upper = c(b, b), V = diag(2), D = c(0, 0))
    in_d <- pmvnorm_fast(upper = c(b, b), V = matrix(0, 2, 2), D = c(1, 1))
    # was 6.1e-305 at b = -4 and exactly 0 at b = -6 for V = I
    expect_lt(abs(as.numeric(in_v) / exact - 1), 1e-10)
    expect_lt(abs(as.numeric(in_d) / exact - 1), 1e-10)
  }
})

test_that("a rectangle that misses the singular support is empty (#410)", {
  V <- matrix(1, 2, 1)                   # X1 = X2 = F exactly
  expect_identical(as.numeric(pmvnorm_fast(lower = c(-Inf, 1),
    upper = c(0, Inf), V = V, D = c(0, 0))), 0)
  expect_identical(as.numeric(pmvnorm_fast(lower = c(-Inf, 0),
    upper = c(0, Inf), V = V, D = c(0, 0))), 0)
  p <- pmvnorm_fast(lower = c(-Inf, -1), upper = c(0, Inf), V = V,
                    D = c(0, 0))
  expect_lt(abs(as.numeric(p) - (pnorm(0) - pnorm(-1))), 1e-13)
})

test_that("an atom on the bound carries its mass (#414)", {
  p <- pmvnorm_fast(lower = c(0, -Inf), upper = c(0, 0), mean = c(0, 0),
                    V = matrix(0, 2, 1), D = c(0, 1))
  expect_lt(abs(as.numeric(p) - 0.5), 1e-15)            # was 0
  expect_identical(as.numeric(pmvnorm_fast(lower = 0, upper = 0, mean = 0,
    V = matrix(0, 1, 1), D = 0)), 1)                    # was 0
  expect_identical(as.numeric(pmvnorm_fast(lower = c(0, -Inf),
    upper = c(0, 0), mean = c(0.5, 0), V = matrix(0, 2, 1),
    D = c(0, 1))), 0)
  slab <- pmvnorm_fast(lower = c(0, -Inf), upper = c(0, 0),
                       V = matrix(0, 2, 1), D = c(1, 1))
  expect_identical(as.numeric(slab), 0)
  expect_equal(attr(slab, "method"), "degenerate-rectangle")
})

test_that("upper-tail cells do not cancel; reflection is exact (#196, #98)", {
  p <- pmvnorm_fast(lower = 9, upper = 10, V = matrix(0, 1, 1), D = 1)
  want <- pnorm(9, lower.tail = FALSE) - pnorm(10, lower.tail = FALSE)
  expect_lt(abs(as.numeric(p) / want - 1), 1e-12)
  p89 <- pmvnorm_fast(lower = 8, upper = 9, V = matrix(0, 1, 1), D = 1)
  w89 <- pnorm(8, lower.tail = FALSE) - pnorm(9, lower.tail = FALSE)
  expect_lt(abs(as.numeric(p89) / w89 - 1), 1e-12)
  for (n in 1:3) {
    up <- pmvnorm_fast(lower = rep(9, n), V = matrix(0, n, 1), D = rep(1, n))
    down <- pmvnorm_fast(upper = rep(-9, n), V = matrix(0, n, 1),
                         D = rep(1, n))
    expect_lt(abs(as.numeric(up) / pnorm(-9)^n - 1), 1e-9)
    expect_lt(abs(as.numeric(up) / as.numeric(down) - 1), 1e-12)
  }
})

test_that("one coordinate is the normal cdf (#395)", {
  sd <- sqrt(1 + 1e-6)
  for (z in c(-4.2, -4.1, -4.0)) {
    p <- pmvnorm_fast(upper = z * sd, V = matrix(1, 1, 1), D = 1e-6)
    expect_lt(abs(as.numeric(p) / pnorm(z) - 1), 1e-12)
  }
})

test_that("units do not change the algorithm (#368)", {
  V <- matrix(c(.5, -.3, .4, .2, -.4, .3), 6, 1)
  D <- c(.8, 1.1, .9, 1.2, 1, .7)
  S <- V %*% t(V) + diag(D)
  mu <- c(.1, -.2, .3, 0, .4, -.1); b <- c(.5, .2, .8, -.1, .4, .6)
  p0 <- pmvnorm_fast(mean = mu, upper = b, sigma = S)
  for (cc in c(1e-12, 1e-6, 1e6, 1e12)) {
    expect_false(is.null(factorize_covariance(cc * S)))   # NULL at 1e-12
    p <- pmvnorm_fast(mean = sqrt(cc) * mu, upper = sqrt(cc) * b,
                      sigma = cc * S)
    expect_equal(attr(p, "method"), "factor")
    expect_lt(abs(as.numeric(p) / as.numeric(p0) - 1), 1e-12)
  }
})

test_that("diagonal and heterogeneous-scale covariance (#132)", {
  D0 <- c(1, 1, 1, 1, 1e8); b <- c(10, 10, 10, 10, -5000)
  expect_equal(ncol(factorize_covariance(diag(D0))$V), 0)
  p <- pmvnorm_fast(upper = b, sigma = diag(D0))
  expect_lt(abs(as.numeric(p) / prod(pnorm(b / sqrt(D0))) - 1), 1e-13)
  for (rho in c(0.5, 0.8, 0.9)) {
    S <- matrix(c(1, rho, 0, rho, 1, 0, 0, 0, 1e16), 3)
    expect_equal(ncol(factorize_covariance(S)$V), 1)    # was rank zero
    p <- pmvnorm_fast(upper = rep(0, 3), sigma = S)
    want <- 0.5 * (0.25 + asin(rho) / (2 * pi))
    expect_lt(abs(as.numeric(p) / want - 1), 1e-6)      # was -41.6% at 0.9
  }
})
