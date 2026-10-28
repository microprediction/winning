# Round-3 port regressions: #408, #319, #659, #100.

test_that("top-k inverse converges on a dominant pair at any n (#408)", {
  for (ne in list(c(3, 6e-4), c(10, 1e-4), c(30, 1e-4))) {
    n <- ne[1]
    p <- c(0.6, 0.4, rep(ne[2], n - 2)); p <- p / sum(p)
    D <- rep(3.2^2, n)
    out <- abilities_from_topk(p, 1, D = D, points = 500, return_info = TRUE)
    expect_true(out$converged)
    expect_lt(out$max_logit_residual, 1e-8)
    q <- top_k_probabilities(out$mu, 1, D = D, points = 500)
    expect_lt(max(abs(q - p)), 1e-6)
  }
  # n_iter = 0 still prices the initializer it returns (#441)
  out0 <- abilities_from_topk(c(0.5, 0.3, 0.2), 1, n_iter = 0,
                              return_info = TRUE)
  expect_equal(out0$iterations, 0L)
  expect_true(is.finite(out0$max_logit_residual))
})

test_that("short D vectors are refused, not recycled (#319)", {
  mu <- c(-2.0, -0.2, 0.4, 1.4)
  expect_error(bottom_k_probabilities(mu, 1, D = c(1, 9)), "per contestant")
  expect_error(bottom_k_probabilities(mu, 1, V = c(.3, -.1, .2, 0), D = c(1, 9),
                                      qa = 5), "per contestant")
  V <- matrix(0, 4, 1)
  nodes <- list(F = matrix(0, 1, 1), W = 1)
  expect_error(win_probabilities_factor(mu, V, c(1, 9), nodes = nodes),
               "per contestant")
  expect_error(abilities_from_probabilities_factor(c(.4, .3, .2, .1), V,
                                                   c(1, 9), nodes = nodes),
               "per contestant")
  # a scalar still broadcasts, and length n is unchanged
  expect_equal(bottom_k_probabilities(mu, 1, D = 2),
               bottom_k_probabilities(mu, 1, D = rep(2, 4)))
  expect_equal(win_probabilities_factor(mu, V, 2, nodes = nodes),
               win_probabilities_factor(mu, V, rep(2, 4), nodes = nodes))
})

test_that("every structure shares the race target contract (#659)", {
  ss <- list(Independent(c(1, 1, 1)),
             Blocks(1:3, c(0, 0, 0), c(1, 1, 1)),
             Nested(1:3, c(0, 0, 0), c(1, 1, 1), coupling = c(0, 0, 0),
                    gamma = 0),
             Tree(c(10, 10, 20), c(.4, .2, .3), c(1, 1, 1), c(3, 3, 0),
                  c(0, 0, .5)))
  for (s in ss) {
    expect_error(abilities_from_race(c(.6, .4, 0), structure = s), "positive")
    expect_error(abilities_from_race(c(.6, .5, -.1), structure = s), "positive")
    expect_error(abilities_from_race(c(.6, NaN, .4), structure = s), "finite")
    expect_error(abilities_from_race(c(.6, Inf, .4), structure = s), "finite")
  }
})

test_that("Nested and Tree inverses are equivariant to units (#100)", {
  mu0 <- c(-1.2, -.35, .15, .55, .85); mu0 <- mu0 - mean(mu0)
  cl <- c(10, 10, 20, 30, 30); L0 <- c(.45, -.2, .3, .55, -.1)
  D0 <- c(.7, 1.1, .8, .9, 1.2); g0 <- c(.35, -.25, .15, .45, -.3)
  pa <- c(4, 4, 5, 5, 0); st0 <- c(0, 0, 0, .5, .35)
  for (kind in c("Nested", "Tree")) {
    ref <- NULL
    for (cc in c(1e-3, 1, 1e3)) {
      s <- if (kind == "Nested") Nested(cl, cc * L0, cc^2 * D0, cc * g0, .7)
           else Tree(cl, cc * L0, cc^2 * D0, pa, cc * st0)
      p <- race_probabilities(cc * mu0, structure = s, points = 257)
      m <- abilities_from_race(p, structure = s, points = 257)
      expect_lt(max(abs(race_probabilities(m, structure = s, points = 257) - p)),
                1e-8)
      if (is.null(ref)) ref <- m / cc else expect_lt(max(abs(m / cc - ref)), 1e-6)
    }
  }
})

test_that("the race inverse and its pair closed form keep units (#100)", {
  p <- c(.5, .3, .2)
  ref <- abilities_from_race(p, D = rep(1, 3), points = 513)
  for (d in c(1e-20, 1e16, 1e20)) {
    m <- abilities_from_race(p, D = rep(d, 3), points = 513)
    expect_lt(max(abs(m / sqrt(d) - ref)), 1e-7)
  }
  # the pair: no 1e-300 variance floor, no overflow of the variance sum
  pp <- c(0.8, 0.2)
  b <- abilities_from_race(pp, D = c(1, 1))
  for (d in c(1e-302, 9e307, 1e308)) {
    m <- abilities_from_race(pp, D = c(d, d))
    expect_true(all(is.finite(m)))
    expect_lt(max(abs(m / sqrt(d) - b)), 1e-12)
  }
})

test_that("the block window has no absolute scale floor (#100)", {
  mu <- c(-1, -.2, .4, 1.1); cl <- c(0, 0, 1, 1)
  L <- c(.4, -.2, .3, -.1); D <- c(.7, 1.2, .8, 1.5)
  ref <- block_race_probabilities(mu, cl, L, D, points = 257, qa = 9)
  for (c in c(1e-13, 1e-14, 1e-20))
    expect_lt(max(abs(block_race_probabilities(c * mu, cl, c * L, c * c * D,
                                               points = 257, qa = 9) - ref)),
              1e-12)
})

test_that("the block inverse is equivariant to units (#100)", {
  p <- c(.5, .3, .2, .1); p <- p / sum(p)
  cl <- c(0, 0, 1, 1); L <- c(.4, -.2, .3, -.1); D <- c(.7, 1.1, .8, .9)
  ref <- abilities_from_block_race(p, cl, L, D)$mu
  for (c in c(1e-6, 1e6, 1e10)) {
    m <- abilities_from_block_race(p, cl, c * L, c^2 * D)$mu
    expect_lt(max(abs(block_race_probabilities(m, cl, c * L, c^2 * D) - p)),
              1e-10)
    expect_lt(max(abs(m / c - ref)), 1e-8)
  }
})
