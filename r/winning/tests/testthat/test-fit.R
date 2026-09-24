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

test_that("a degraded cov= fit is announced where it cannot be routed", {
  # The normal race with no slopes is routed to GHK (see below). Every
  # other call needs the factor form -- slopes for the inverse's
  # preconditioner, a non-normal base -- and prices the degraded fit, so
  # it says so. It used to say nothing at all on any path, which is the
  # one failure mode that looks like an answer.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.05 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.6, 0.6, length.out = n)
  expect_warning(race_probabilities(mu, cov = C, return_slopes = TRUE),
                 "degraded")
  expect_warning(abilities_from_race(c(.4, .2, .1, .1, .07, .06, .04, .03),
                                     cov = C, base = "gumbel"), "degraded")
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

test_that("a degraded cov= fit is routed to GHK, forward and inverse", {
  # python routes this case rather than pricing the fit; so does this port
  # now, to the same canonical runner order and in log space throughout.
  # Routing only the forward would make the two front doors describe
  # different races, which is python's #164.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.05 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.6, 0.6, length.out = n)
  expect_silent(p <- race_probabilities(mu, cov = C))   # routed: no warning
  expect_equal(sum(p), 1, tolerance = 1e-12)
  mu_back <- abilities_from_race(p, cov = C)
  expect_equal(race_probabilities(mu_back, cov = C), p, tolerance = 1e-7)
  expect_equal(mean(mu_back), 0, tolerance = 1e-9)
})

test_that("GHK is permutation-equivariant and its slopes have the right sign", {
  set.seed(11)
  n <- 6
  A <- matrix(rnorm(n * n), n, n)
  S <- A %*% t(A) + n * diag(n)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- rnorm(n)
  g <- .ghk_race(mu, C, want_slopes = TRUE)
  expect_equal(sum(g$p), 1, tolerance = 1e-12)
  expect_true(all(g$dlogp < 0))       # min-wins: a higher mu loses mass
  perm <- c(4, 1, 6, 2, 5, 3)
  gp <- .ghk_race(mu[perm], C[perm, perm])$p
  back <- numeric(n); back[perm] <- gp
  expect_equal(back, g$p, tolerance = 1e-12)
})
