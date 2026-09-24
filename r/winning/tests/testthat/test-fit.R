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

test_that("a degraded cov= fit is announced, not priced silently", {
  # The python package routes this case to scrambled-Sobol GHK; this port
  # has none and prices the fit, 4.6e-3 away on the n=8 exactly-rank-3
  # correlation. It used to say nothing at all, which is the one failure
  # mode that looks like an answer.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.05 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.6, 0.6, length.out = n)
  expect_warning(race_probabilities(mu, cov = C), "degraded")
  expect_warning(abilities_from_race(c(.4, .2, .1, .1, .07, .06, .04, .03),
                                     cov = C), "degraded")
  fit <- fit_covariance(C)
  expect_true(fit$degraded)
  expect_true(fit$rank <= n)          # #180: the eigen arm used to ask for
                                      # more columns than there are
})

test_that("a healthy cov= fit is priced without a warning", {
  set.seed(7)
  n <- 20
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.3 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.8, 0.8, length.out = n)
  expect_silent(p <- race_probabilities(mu, cov = C, points = 257))
  expect_equal(sum(p), 1, tolerance = 1e-9)
  expect_false(fit_covariance(C)$degraded)
})
