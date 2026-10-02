# Factor-core batch, R parity with python tests/test_factor_core_issues.py

test_that("the top-k window is scale-free (#370)", {
  mu <- c(-.5, .2, .8, -.1); sd <- c(.7, 1.1, .9, 1.3)
  for (cc in c(1e-18, 1e9))
    expect_equal(top_k_probabilities(cc * mu, 2, D = (cc * sd)^2),
                 top_k_probabilities(mu, 2, D = sd^2), tolerance = 1e-9)
})

test_that("bottom-k keeps a rare last place (#365)", {
  b <- bottom_k_probabilities(c(-12, 0, 0), 1, D = rep(1, 3))
  expect_equal(b[1] / 7.726967753938468796e-24, 1, tolerance = 1e-6)
})

test_that("a ridge stall on an impossible board is not convergence (#353, #105)", {
  for (ridge in c(0, 1e-4, 0.0025)) {
    f <- suppressWarnings(loc_scale_from_topk_pair(
      c(.8, .1, .1), 1, c(.2, .9, .9), 2, points = 1025, ridge = ridge,
      return_info = TRUE))
    expect_false(f$converged)
    expect_false(f$nested)
  }
})

test_that("an exact warm start returns the canonical gauge (#360)", {
  mu <- c(-3, -1, 1, 3); sd <- c(2, 3, 4, 5)
  q1 <- top_k_probabilities(mu, 1, D = sd^2, points = 1025)
  q2 <- top_k_probabilities(mu, 2, D = sd^2, points = 1025)
  w <- loc_scale_from_topk_pair(q1, 1, q2, 2, D0 = sd^2, mu0 = mu,
                                points = 1025, return_info = TRUE)
  expect_true(w$converged)
  expect_equal(exp(mean(log(w$sd))), 1, tolerance = 1e-12)
  expect_equal(mean(w$mu), 0, tolerance = 1e-12)
})

test_that("fractional ranks are refused (#317)", {
  for (r in list(1.5, 1.999, 2.0001, NaN, Inf))
    expect_error(abilities_from_rank_marginal(c(.4, .3, .2, .1), r),
                 "whole-number rank")
})

test_that("the odd-field middle rank needs mu0 (#378)", {
  P <- rank_probabilities(c(-1, -.4, .05, .45, .9), D = rep(1, 5),
                          points = 257)
  expect_error(abilities_from_rank_marginal(P[, 3], 3, D = rep(1, 5),
                                            points = 257), "mu0")
})

test_that("top-k factor nodes adapt to sharpness (#340)", {
  mu <- c(-.15, .05, .1, 0); V <- matrix(c(-3, -1, 1, 3), ncol = 1)
  D <- rep(.01, 4)
  expect_equal(top_k_probabilities(mu, 1, V = V, D = D, points = 1025),
               race_probabilities(mu, V = V, D = D, points = 1025),
               tolerance = 1e-6)
})

test_that("abilities_from_topk is unit-equivariant (#100)", {
  mu0 <- c(-.8, -.3, 0, .4, .7)
  q <- top_k_probabilities(mu0, 2)
  for (cc in c(1e-3, 1e3)) {
    m <- abilities_from_topk(q, 2, D = rep(cc^2, 5), return_info = TRUE)
    expect_true(m$converged)
    expect_equal(m$mu / cc, mu0 - mean(mu0), tolerance = 1e-6)
  }
})
