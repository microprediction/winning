# Run by R CMD check through tests/testthat.R, so the package is
# already attached and its internals are in scope. These files used
# to source(file.path("..", "..", "R", ...)) instead, which only
# works from the repo and fails inside a check -- which is how they
# went unrun (#142).
test_that("synthetic MNP fits, converges, recovers slope direction", {
  set.seed(9)
  J <- 3; Tn <- 600
  V_true <- rbind(0, matrix(c(0.9, 0, 0.5, 0.7), 2, 2))
  x <- rnorm(Tn * J)
  alt <- rep(1:J, Tn)
  Xint <- matrix(0, Tn * J, 2)
  for (j in 2:J) Xint[alt == j, j - 1] <- 1
  mu <- matrix(cbind(Xint, x) %*% c(0.3, -0.2, 0.9), Tn, J, byrow = TRUE)
  eps <- t(V_true %*% matrix(rnorm(2 * Tn), 2)) +
    matrix(rnorm(Tn * J), Tn, J)
  choice <- max.col(mu + eps)
  df <- data.frame(id = rep(1:Tn, each = J), alt = alt,
                   chosen = as.integer(
                     sequence(rep(J, Tn)) == rep(choice, each = J)),
                   price = x)
  fit <- rprobit_fast(df, "price", maxit = 200)
  expect_equal(fit$convergence, 0)
  expect_gt(fit$coefficients["price"], 0.5)
  expect_false(fit$boundary)
})

test_that("a malformed choice set is rejected, not reshaped", {
  # The reshape downstream is POSITIONAL, so row order alone decides
  # which alternative a row is priced as. A count check cannot see a
  # duplicate paired with an omission: the observation still has J rows,
  # `nrow(df) == Tn * J` passed, and the duplicate was priced as the
  # missing alternative, returning an ordinary-looking fit (#230).
  d <- data.frame(id = c(1, 1, 1, 2, 2, 2),
                  alt = c(1, 1, 3, 1, 2, 3),   # id 1 duplicates 1, omits 2
                  chosen = c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE),
                  x = c(0, 10, 0, 0, 0, 0))
  expect_error(rprobit_fast(d, covariates = "x", r = 1L, Qf = 3L, Qz = 3L,
                            maxit = 1L),
               "exactly once")
  # the message must name the observation and the alternative
  msg <- tryCatch(rprobit_fast(d, covariates = "x", r = 1L, Qf = 3L,
                               Qz = 3L, maxit = 1L),
                  error = function(e) conditionMessage(e))
  expect_match(msg, "observation '1'")
  expect_match(msg, "alternative '1'")
})

test_that("a well-formed panel is untouched", {
  set.seed(4)
  n <- 60L; J <- 3L
  df <- data.frame(id = rep(seq_len(n), each = J),
                   alt = rep(1:J, times = n), x = rnorm(n * J))
  u <- 0.7 * df$x + rnorm(n * J)
  df$chosen <- unlist(lapply(split(u, df$id),
                             function(v) as.integer(seq_along(v) == which.max(v))))
  fit <- rprobit_fast(df, covariates = "x", r = 1L, Qf = 5L, Qz = 5L,
                      maxit = 40L)
  expect_true(is.finite(fit$coefficients[["x"]]))
  expect_gt(fit$coefficients[["x"]], 0)
})

test_that("a binary choice fits instead of running off the end of V", {
  # The triangular-loading loop read `(col + 1L):J`, and at J = 2 with the
  # default r = 2L the col = 2 pass evaluates 3:2 -- which in R is the
  # two-element vector c(3, 2), not empty. The first assignment was
  # V[3, 2] on a 2x2 matrix, so every binary-choice fit died with
  # "subscript out of bounds" (#183). nw counts one free loading there, so
  # wfree[2] was out of range too. (The default is now min(2, J - 2) = 0
  # at J = 2, since any loading is unidentified there (#201); the loop's
  # J = 2 behaviour is pinned directly in the next test.)
  set.seed(11)
  n <- 120L; J <- 2L
  id <- rep(seq_len(n), each = J)
  alt <- rep(1:J, times = n)
  x <- rnorm(n * J)
  u <- 0.8 * x + rnorm(n * J)
  chosen <- unlist(lapply(split(u, id),
                          function(v) as.integer(seq_along(v) == which.max(v))))
  df <- data.frame(id = id, alt = alt, x = x, chosen = chosen)
  fit <- rprobit_fast(df, covariates = "x", maxit = 60L)
  expect_true(is.finite(fit$coefficients[["x"]]))
  expect_gt(fit$coefficients[["x"]], 0)        # the true slope is +0.8
})

test_that("the triangular loop is empty exactly when it should be", {
  # `(col + 1L):J` counts DOWN when col + 1 > J; seq_len does not.
  visited <- function(J, r) {
    out <- character(0)
    for (col in seq_len(r)) for (row in col + seq_len(J - col))
      out <- c(out, sprintf("%d,%d", row, col))
    out
  }
  expect_equal(visited(2L, 2L), "2,1")          # one free loading, in range
  # J = 5, r = 2: col 1 fills rows 2..5 and col 2 fills rows 3..5
  expect_equal(length(visited(5L, 2L)), (5L - 1L) + (5L - 2L))
  expect_true(all(vapply(strsplit(visited(5L, 2L), ","),
                         function(p) as.integer(p[1]) <= 5L, TRUE)))
  expect_equal(visited(1L, 1L), character(0))   # nothing to fill
})

# --- every observation is accounted for (#194) ------------------------
#
# .nll_core iterates over the LEGAL labels and gathers the rows matching
# each, so a row whose choice is outside 1..J is never visited: logp
# stays 0 there, which adds zero negative log-likelihood and zero
# gradient, and the fit silently optimised a SUBSET while reporting it
# as the whole. NA rows did the same. Dropping observations RAISES the
# log-likelihood, because there is less of it, so nothing downstream can
# tell it from a better fit.

test_that("a choice outside the alternatives is refused", {
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  th <- c(0.5, 0.2, 0.3)
  nd <- .halton_nodes3(1L, m = 5L)
  for (bad in list(9L, 0L, -1L, NA_integer_)) {
    ch <- rep(1:3, length.out = 30)
    ch[1] <- bad
    expect_error(.nll_core(th, list(X), ch, 3L, 1L, nd, nd,
                           want_grad = FALSE),
                 "not an alternative in 1\\.\\.3")
  }
})

test_that("a choice vector of the wrong length is refused", {
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  th <- c(0.5, 0.2, 0.3)
  nd <- .halton_nodes3(1L, m = 5L)
  for (n in c(20L, 31L)) {
    expect_error(.nll_core(th, list(X), rep(1:3, length.out = n),
                           3L, 1L, nd, nd, want_grad = FALSE),
                 "one entry per observation")
  }
})

test_that("the message names the first offender and counts them", {
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  nd <- .halton_nodes3(1L, m = 5L)
  ch <- rep(1:3, length.out = 30); ch[4] <- 9L
  msg <- tryCatch(.nll_core(c(0.5, 0.2, 0.3), list(X), ch, 3L, 1L,
                            nd, nd, want_grad = FALSE),
                  error = function(e) conditionMessage(e))
  expect_match(msg, "choice\\[4\\] = 9")
  expect_match(msg, "1 of 30")
})

test_that("a valid panel is untouched", {
  set.seed(2)
  n <- 40L; J <- 3L
  df <- data.frame(id = rep(seq_len(n), each = J),
                   alt = rep(1:J, times = n), x = rnorm(n * J))
  u <- 0.6 * df$x + rnorm(n * J)
  df$chosen <- unlist(lapply(split(u, df$id),
                             function(v) as.integer(seq_along(v) == which.max(v))))
  fit <- rprobit_fast(df, covariates = "x", r = 1L, Qf = 5L, Qz = 5L,
                      maxit = 10L)
  expect_true(is.finite(fit$coefficients[["x"]]))
})

test_that("fractional, infinite and NaN labels are refused; 1.0 is fine", {
  # 1.5 is inside 1..J but matches no `choice == k_alt` bucket, so it was
  # dropped in silence exactly like an out-of-range label (#194)
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  th <- c(0.5, 0.2, 0.3)
  nd <- .halton_nodes3(1L, m = 5L)
  for (bad in list(1.5, 2.0001, Inf, NaN)) {
    ch <- as.numeric(rep(1:3, length.out = 30))
    ch[1] <- bad
    expect_error(.nll_core(th, list(X), ch, 3L, 1L, nd, nd,
                           want_grad = FALSE),
                 "not an alternative in 1\\.\\.3")
  }
  chi <- rep(1:3, length.out = 30)
  th <- c(0.5, 0.2, 0.3, 0.1)              # 2 betas + 2 loadings
  a <- .nll_core(th, list(X), chi, 3L, 1L, nd, nd, want_grad = FALSE)
  b <- .nll_core(th, list(X), as.numeric(chi), 3L, 1L, nd, nd,
                 want_grad = FALSE)
  expect_equal(a$value, b$value)
})

# --- the factor rank must be identified (#201) --------------------------
#
# With unit idiosyncratic variances the free loadings number
# r*J - r*(r+1)/2 against J*(J-1)/2 - 1 shape degrees of freedom of the
# differenced covariance; at r >= J - 1 the scale is free and
# (2 beta, V') prices every choice like (beta, V).
test_that("the contrast covariance ridge is exact at J = 3, r = 2", {
  A <- rbind(c(-1, 1, 0), c(-1, 0, 1))
  V <- rbind(c(0, 0), c(.5, 0), c(.2, .7))
  Vp <- rbind(c(0, 0), c(2.6457513110645907, 0),
              c(1.2850792082313727, 2.543338638202044))
  C <- function(V) A %*% (V %*% t(V) + diag(3)) %*% t(A)
  expect_equal(C(Vp), 4 * C(V), tolerance = 1e-12)
})

test_that("ranks above J - 2 are refused and the default is identified", {
  for (J in 2:5) {
    expect_identical(.check_rank(NULL, J), as.integer(min(2L, J - 2L)))
    for (r in 0:(J - 2L)) expect_identical(.check_rank(r, J), as.integer(r))
    for (r in c(J - 1L, J)) expect_error(.check_rank(r, J), "not identified")
  }
  set.seed(3)
  n <- 40L; J <- 3L
  df <- data.frame(id = rep(seq_len(n), each = J),
                   alt = rep(1:J, times = n), x = rnorm(n * J))
  u <- 0.6 * df$x + rnorm(n * J)
  df$chosen <- unlist(lapply(split(u, df$id),
                             function(v) as.integer(seq_along(v) == which.max(v))))
  expect_error(rprobit_fast(df, covariates = "x", r = 2L, maxit = 5L),
               "not identified")
  fit <- rprobit_fast(df, covariates = "x", Qf = 5L, Qz = 5L, maxit = 20L)
  expect_identical(fit$r, 1L)
})
