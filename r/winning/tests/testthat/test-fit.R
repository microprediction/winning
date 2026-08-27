test_that("fit_covariance recovers in-grammar truths", {
  set.seed(1)
  n <- 30
  V <- matrix(rnorm(n * 2), n, 2) * 0.5
  D <- 0.5 + runif(n)
  C <- V %*% t(V) + diag(D)
  mu <- sort(rnorm(n))
  fit <- fit_covariance(C)
  p1 <- race_probabilities(mu, cov = C)
  p0 <- race_probabilities(mu, V = V, D = D, F = fit$F[, 1:2] * 0 + 
                             .halton_normal_nodes(2, 2048)$F,
                           W = .halton_normal_nodes(2, 2048)$W)
  expect_lt(0.5 * sum(abs(p1 - p0)), 2e-3)
})

test_that("cov= inversion round trip", {
  set.seed(2)
  n <- 20
  V <- matrix(rnorm(n * 2), n, 2) * 0.5
  C <- V %*% t(V) + diag(0.5 + runif(n))
  mu0 <- sort(rnorm(n)); mu0 <- mu0 - mean(mu0)
  p <- race_probabilities(mu0, cov = C)
  mu_hat <- abilities_from_race(p, cov = C)
  expect_lt(max(abs(mu_hat - mu0)), 1e-4)
})
