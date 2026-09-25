source(file.path("..", "..", "R", "pmvnorm_fast.R"))

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

# --- explicit factor rank past the old prime table (#143) --------------
#
# .halton_unit tabulated exactly six primes and indexed them by rank, so
# an explicit V of rank 7 selected primes[7] = NA and the Halton loop
# died with a missing-value error. pmvnorm_fast documents no rank limit,
# and the python reference has none: it uses arbitrary-dimensional Sobol
# nodes. The primes are generated now.

test_that("an explicit factor rank past six is accepted", {
  for (r in c(6L, 7L, 8L, 12L)) {
    n <- r + 1L
    V <- diag(n)[, seq_len(r), drop = FALSE] * 0.4
    p <- pmvnorm_fast(upper = rep(0, n), V = V, D = rep(1, n))
    expect_true(is.finite(as.numeric(p)), info = paste("rank", r))
    expect_gt(as.numeric(p), 0)
    expect_lt(as.numeric(p), 1)
  }
})

test_that("a rank past six is no less accurate than rank six was", {
  # V = 0.4 * the first r columns of I_n leaves every coordinate
  # independent with mean 0, so P(X <= 0) = 0.5^n exactly and the gap is
  # pure node error. Halton in r dimensions loses accuracy as r grows,
  # and the ranks that used to be refused are in the same band as rank 6,
  # which always worked -- measured relative error 1.1e-3 at rank 6,
  # 9.1e-4 at 7, 1.5e-3 at 8, 3.4e-3 at 12. The point of this test is
  # that rank 7 is ordinary, not that it is exact.
  for (r in c(6L, 7L, 8L, 12L)) {
    n <- r + 1L
    V <- diag(n)[, seq_len(r), drop = FALSE] * 0.4
    p <- as.numeric(pmvnorm_fast(upper = rep(0, n), V = V, D = rep(1, n)))
    expect_lt(abs(p - 0.5^n) / 0.5^n, 5e-3, label = paste("rank", r))
  }
})

test_that(".first_primes generates the primes it used to tabulate", {
  expect_equal(.first_primes(6), c(2, 3, 5, 7, 11, 13))
  expect_equal(.first_primes(1), 2)
  expect_equal(length(.first_primes(0)), 0)
  expect_equal(tail(.first_primes(20), 1), 71)
})
