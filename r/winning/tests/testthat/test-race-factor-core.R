# Factor-core batch, R parity with python tests/test_factor_core_issues.py

test_that("weights normalise without overflow (#263)", {
  mu <- c(0, 0.3, 1); V <- matrix(c(0, 1, -0.5), ncol = 1)
  F <- matrix(c(-1, 1), ncol = 1)
  a <- race_probabilities(mu, V = V, D = rep(1, 3), F = F, W = c(1e307, 1e307))
  b <- race_probabilities(mu, V = V, D = rep(1, 3), F = F, W = c(1e308, 1e308))
  expect_equal(a, b, tolerance = 1e-12)
})

test_that("a lattice needs two points (#444)", {
  for (pts in list(0, 1, 2.5, NaN))
    expect_error(race_probabilities(c(-.5, 0, .5), D = rep(1, 3),
                                    points = pts), "points")
})

test_that("non-finite targets are refused and pairs use the longshot (#110)", {
  expect_error(abilities_from_race(c(.8, NaN)), "finite")
  expect_error(abilities_from_race(c(Inf, 1)), "finite")
  mu <- abilities_from_race(c(1, 1e-17), D = c(1, 1))
  expect_true(all(is.finite(mu)))
  # min-wins: runner 2 wins with Phi((mu1 - mu2) / sqrt(2))
  expect_equal(pnorm((mu[1] - mu[2]) / sqrt(2), log.p = TRUE),
               log(1e-17 / (1 + 1e-17)), tolerance = 1e-10)
})

test_that("structure= honours controls and refuses a second covariance (#89)", {
  s <- Independent(rep(1, 3))
  expect_error(race_probabilities(c(0, .3, .6), structure = s, D = rep(1, 3)),
               "structure")
  expect_error(abilities_from_race(c(.5, .3, .2), structure = s, V = rep(1, 3)),
               "structure")
  a <- suppressWarnings(abilities_from_race(c(.5, .3, .2), structure = s,
                                            points = 257, n_iter = 1,
                                            tol = 1e-1))
  b <- suppressWarnings(abilities_from_race(c(.5, .3, .2), structure = s,
                                            points = 257, n_iter = 999,
                                            tol = 1e-99))
  expect_false(identical(a, b))
  mu <- c(-1.88, 3.24, -0.27, 0.06); D <- c(0.056, 1.70, 14.8, 3.60)
  expect_equal(race_probabilities(mu, structure = Independent(D), points = 65,
                                  window = "span"),
               race_probabilities(mu, D = D, points = 65, window = "span"))
  bl <- Blocks(c(0, 0, 1, 1), c(.3, .2, .4, .1), rep(1, 4))
  expect_error(race_probabilities(c(-.4, -.1, .2, .5), structure = bl,
                                  base = "logistic"), "base")
  expect_error(abilities_from_race(c(.4, .3, .2, .1), structure = bl,
                                   base = "gumbel"), "base")
})

test_that("the factor core validates its law and gauge (#263, #416, #444, #70)", {
  mu <- c(0, 0.3, 1); V <- matrix(c(0, 1, -0.5), ncol = 1); D <- rep(1, 3)
  F <- matrix(c(-5, 5), ncol = 1)
  expect_error(win_probabilities_factor(mu, V, D, nodes = list(F = F, W = c(2, -1))),
               "negative")
  expect_error(win_probabilities_factor(mu, V, D, nodes = list(F = F, W = c(0, 0))),
               "positive")
  expect_error(win_probabilities_factor(mu, V, D, points = 1), "points")
  mz <- c(.1, .2, .4); Vz <- matrix(c(2, 0, -2), ncol = 1); Dz <- rep(1e-6, 3)
  centre <- win_probabilities_factor(mz, Vz, Dz, nodes = list(F = matrix(0, 1, 1), W = 1))
  padded <- win_probabilities_factor(mz, Vz, Dz,
    nodes = list(F = matrix(c(-100, 0, 100), ncol = 1), W = c(0, 1, 0)))
  expect_equal(padded, centre, tolerance = 1e-12)
  mu0 <- c(-1, -.2, .3, .9); Vg <- matrix(c(.2, -.1, .3, -.4), ncol = 1)
  p <- win_probabilities_factor(mu0, Vg, rep(1, 4), points = 1001)
  a <- abilities_from_probabilities_factor(p, Vg, rep(1, 4), points = 1001, tol = 1e-8)
  b <- abilities_from_probabilities_factor(p, Vg + 100, rep(1, 4), points = 1001, tol = 1e-8)
  expect_equal(a, b, tolerance = 1e-6)
})
