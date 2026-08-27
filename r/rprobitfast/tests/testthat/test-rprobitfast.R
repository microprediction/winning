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
