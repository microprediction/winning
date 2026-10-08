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
  nd <- .sobol_nodes3(1L, m = 5L)
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
  nd <- .sobol_nodes3(1L, m = 5L)
  for (n in c(20L, 31L)) {
    expect_error(.nll_core(th, list(X), rep(1:3, length.out = n),
                           3L, 1L, nd, nd, want_grad = FALSE),
                 "one entry per observation")
  }
})

test_that("the message names the first offender and counts them", {
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  nd <- .sobol_nodes3(1L, m = 5L)
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

# --- shared-engine regressions (identical in mlogitfast and rprobitfast) ---

test_that("sharp-regime nodes exist at every rank the fitters accept (#388)", {
  for (r in 1:6) {
    nd <- .sobol_nodes3(r, 5L)
    expect_equal(dim(nd$F), c(32L, r + 1L))
    expect_true(all(is.finite(nd$F)) && all(is.finite(nd$W)))
    expect_equal(sum(nd$W), 1)
  }
  # the rule is built lazily, once
  lz <- .lazy_sharp_nodes(4L, 5L)
  expect_identical(lz(), lz())
  expect_equal(ncol(lz()$F), 5L)
})

test_that("node dispatch uses the centred pairwise-safe bound (#419)", {
  V <- matrix(c(0, 1, -2.7), ncol = 1)
  expect_equal(.mnp_sharpness(V), sqrt(2) * max(abs(V - mean(V))))
  expect_gt(.mnp_sharpness(V), 3)               # raw max |V_i| is 2.7
  # gauge invariance: V -> V + 1 c'
  V2 <- matrix(c(0, 1, -2.7, 0.3, -0.4, 2), ncol = 2)
  expect_equal(.mnp_sharpness(V2), .mnp_sharpness(sweep(V2, 2, c(5, -3), "+")))
  # the #419 fixture: alternative 3 wins with exact bivariate-normal
  # probability 0.15149432190016066; Hermite-7 said 0.16517
  X <- diag(3)
  theta <- c(-3.25, 2.47, -1.61, 1, -2.7)
  gh <- .nodes3(7L, 7L, 1L)
  got <- exp(-.nll_core(theta, list(X), 3L, 3L, 1L, nodes = gh,
                        nodes_sharp = .sobol_nodes3(1L, 10L))$value)
  expect_lt(abs(got - 0.15149432190016066), 2e-3)
  forced_gh <- exp(-.nll_core(theta, list(X), 3L, 3L, 1L, nodes = gh,
                              nodes_sharp = gh)$value)
  expect_gt(abs(forced_gh - 0.15149432190016066), 1e-2)
})

test_that("choices are keyed by observation, one per id (#324, #435)", {
  ids <- rep(1:3, each = 3); alt <- rep(1:3, 3)
  ok <- c(TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE)
  expect_equal(.choices_by_id(ids, alt, ok), c(1L, 2L, 3L))
  expect_equal(.choices_by_id(ids, alt, as.integer(ok)), c(1L, 2L, 3L))
  cancel <- c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE)
  expect_error(.choices_by_id(ids, alt, cancel, c("a", "b", "c")),
               "observation 'a' has 2 chosen rows")
  none <- ok; none[5] <- FALSE
  expect_error(.choices_by_id(ids, alt, none), "observation '2' has 0")
  two <- ok; two[6] <- TRUE
  expect_error(.choices_by_id(ids, alt, two), "has 2 chosen")
  na <- ok; na[4] <- NA
  expect_error(.choices_by_id(ids, alt, na), "missing")
  expect_error(.choices_by_id(ids, alt, ok * 2), "logical or 0/1")
})

test_that("rank-four fits reach the likelihood (#388)", {
  for (J in c(5L, 6L)) {
    df <- expand.grid(alt = seq_len(J), id = 1:3)
    df$chosen <- df$alt == ((df$id - 1L) %% J + 1L)
    df$x <- seq_len(nrow(df)) / nrow(df)
    # every identified rank (r <= J - 2, #201) reaches the likelihood;
    # J = 5, r = 4 is refused by that contract, not by the node table
    for (r in seq(3L, J - 2L)) {
      fit <- rprobit_fast(df, "x", r = r, Qf = 3L, Qz = 3L, maxit = 1L)
      expect_true(is.finite(fit$logLik))
    }
  }
  expect_error(rprobit_fast(df[df$alt <= 5L, ], "x", r = 4L, Qf = 3L,
                            Qz = 3L, maxit = 1L), "not identified")
})

test_that("cancelling duplicate/missing choices are refused (#324, #435)", {
  d <- data.frame(id = rep(1:3, each = 3), alt = rep(1:3, 3),
                  x = c(-1, .2, .7, .4, -1.2, .1, .8, -.3, 1.1))
  d$chosen <- c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE)
  expect_error(rprobit_fast(d, "x", r = 1L, Qf = 3L, Qz = 3L, maxit = 1L),
               "observation '1' has 2 chosen rows")
  d$chosen <- c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE)
  expect_error(rprobit_fast(d, "x", r = 1L, Qf = 3L, Qz = 3L, maxit = 1L),
               "exactly one")
  d$chosen <- c(TRUE, FALSE, FALSE, NA, TRUE, FALSE, FALSE, FALSE, TRUE)
  expect_error(rprobit_fast(d, "x", r = 1L, Qf = 3L, Qz = 3L, maxit = 1L),
               "missing")
  # a valid row permutation keeps every choice attached to its id
  d$chosen <- c(TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE)
  f1 <- rprobit_fast(d, "x", r = 1L, Qf = 5L, Qz = 5L, maxit = 20L)
  p <- c(9, 4, 1, 8, 2, 6, 3, 7, 5)
  f2 <- rprobit_fast(d[p, ], "x", r = 1L, Qf = 5L, Qz = 5L, maxit = 20L)
  expect_equal(f1$logLik, f2$logLik)
  expect_equal(f1$coefficients, f2$coefficients)
})

test_that("fractional, infinite and NaN labels are refused; 1.0 is fine", {
  # 1.5 is inside 1..J but matches no `choice == k_alt` bucket, so it was
  # dropped in silence exactly like an out-of-range label (#194)
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  th <- c(0.5, 0.2, 0.3)
  nd <- .sobol_nodes3(1L, m = 5L)
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

test_that("unused factor levels are not phantom observations or alternatives (#464)", {
  set.seed(4)
  n <- 30L; J <- 3L
  base <- data.frame(id = rep(seq_len(n), each = J),
                     alt = rep(c("x", "y", "z"), times = n),
                     v = rnorm(n * J))
  u <- 0.7 * base$v + rnorm(n * J)
  base$chosen <- unlist(lapply(split(u, base$id),
                               function(w) seq_along(w) == which.max(w)))
  fit <- function(d) rprobit_fast(d, "v", r = 1L, Qf = 3L, Qz = 3L,
                                  maxit = 5L)
  # unused middle and trailing id levels
  d <- base
  d$id <- factor(d$id * 2L, levels = seq_len(2L * n + 1L))
  ref <- fit(droplevels(d))
  got <- fit(d)
  expect_equal(got$coefficients, ref$coefficients)
  expect_equal(got$logLik, ref$logLik)
  # unused middle and trailing alternative levels
  d <- base
  d$alt <- factor(d$alt, levels = c("x", "w", "y", "z", "zz"))
  ref <- fit(droplevels(d))
  got <- fit(d)
  expect_identical(got$J, 3L)
  expect_equal(got$coefficients, ref$coefficients)
  expect_equal(got$logLik, ref$logLik)
})
