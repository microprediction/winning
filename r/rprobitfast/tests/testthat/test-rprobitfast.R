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
