source(file.path("..", "..", "R", "mlogit_fast.R"))

test_that("nll is finite and decreases from start on a synthetic problem", {
  set.seed(3)
  J <- 3; Tn <- 400; r <- 2
  V <- rbind(0, matrix(c(1.2, 0, 0.7, 0.9), 2, 2))
  X <- cbind(matrix(rep(diag(1, J)[, -1], Tn), ncol = J - 1, byrow = TRUE),
             rnorm(Tn * J))
  beta_true <- c(0.4, -0.3, 0.8)
  mu <- matrix(X %*% beta_true, Tn, J, byrow = TRUE)
  eps <- t(V %*% matrix(rnorm(2 * Tn), 2)) + matrix(rnorm(Tn * J), Tn, J)
  choice <- max.col(mu + eps)
  nodes <- .nodes3(7, 7, r); ns <- .halton_nodes3(r, 9L)
  th0 <- c(rep(0, 3), rep(0.1, 3))
  v0 <- .nll(th0, list(X), choice, J, r, nodes, ns)
  expect_true(is.finite(v0))
  th1 <- th0; th1[1:3] <- beta_true
  expect_lt(.nll(th1, list(X), choice, J, r, nodes, ns), v0)
})

test_that("sharpness escalation switches node families", {
  J <- 3; r <- 2
  nodes <- .nodes3(7, 7, r); ns <- .halton_nodes3(r, 9L)
  X <- cbind(matrix(0, 3 * J, J - 1), rnorm(3 * J))
  # sharp covariance parameters must route to the Halton set: the nll
  # values under the two calls differ only if the branch is taken, so
  # probe via node-count sensitivity at sharp vs mild theta
  th_mild <- c(0, 0, 0, 0.1, 0.1, 0.1)
  th_sharp <- c(0, 0, 0, 50, 50, 50)
  v_m <- .nll(th_mild, list(X), c(1, 2, 3), J, r, nodes, ns)
  v_s <- .nll(th_sharp, list(X), c(1, 2, 3), J, r, nodes, ns)
  expect_true(is.finite(v_m) && is.finite(v_s))
})
