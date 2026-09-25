source(file.path("..", "..", "R", "engine.R"))
source(file.path("..", "..", "R", "rprobit_fast.R"))

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

test_that("a binary choice fits instead of running off the end of V", {
  # The triangular-loading loop read `(col + 1L):J`, and at J = 2 with the
  # default r = 2L the col = 2 pass evaluates 3:2 -- which in R is the
  # two-element vector c(3, 2), not empty. The first assignment was
  # V[3, 2] on a 2x2 matrix, so every binary-choice fit died with
  # "subscript out of bounds" (#183). nw counts one free loading there, so
  # wfree[2] was out of range too.
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
