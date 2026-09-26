
# --- W and c*W are the same factor law (#281's defect, R's copy) ------
#
# The forward divides its accumulated shares by their total, so it was
# invariant either way. race_jacobian does not, so J(W) and J(10W)
# differed by 1.473 -- a Newton step scaled by the spelling of the law
# rather than the law. Same defect as #281 in the standalone javascript
# module and #290 in the browser tree; python has never had it, because
# _setup puts W through winning.shapes.as_weights.

test_that("the forward and the jacobian are invariant to W -> cW", {
  mu <- c(-0.6, -0.2, 0.15, 0.7)
  V <- matrix(c(1.2, -0.4, 0.6, -1.0, -0.7, 1.1, 0.9, -0.5), 4, 2)
  D <- c(0.5, 0.8, 0.6, 0.9)
  F <- matrix(c(-1, -1, 1, 1, -1, 1, -1, 1), 4, 2)
  W <- rep(0.25, 4)
  p0 <- race_probabilities(mu, V = V, D = D, F = F, W = W)
  j0 <- race_jacobian(mu, V = V, D = D, F = F, W = W)
  for (cc in c(1e-6, 0.1, 10, 1e6)) {
    expect_lt(max(abs(race_probabilities(mu, V = V, D = D, F = F,
                                         W = W * cc) - p0)), 1e-15)
    expect_lt(max(abs(race_jacobian(mu, V = V, D = D, F = F,
                                    W = W * cc) - j0)), 1e-15)
  }
  # an unnormalised spelling of the SAME law, not a rescaling
  expect_lt(max(abs(race_probabilities(mu, V = V, D = D, F = F,
                                       W = c(1, 1, 1, 1)) - p0)), 1e-15)
})

test_that("a W that is not a law is refused", {
  mu <- c(-0.6, -0.2, 0.15, 0.7)
  V <- matrix(c(1.2, -0.4, 0.6, -1.0, -0.7, 1.1, 0.9, -0.5), 4, 2)
  D <- c(0.5, 0.8, 0.6, 0.9)
  F <- matrix(c(-1, -1, 1, 1, -1, 1, -1, 1), 4, 2)
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = c(0.5, -0.5, 0.5, 0.5)),
               "non-negative")
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = rep(0, 4)),
               "positive total")
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = c(0.25, NA, 0.25, 0.25)),
               "positive total|finite")
})

test_that("factor nodes carry the loadings' rank", {
  # F with the wrong number of ROWS was accepted and priced a different
  # quadrature outright; too few weights returned all NA; extra weights
  # were ignored. Same contract as the browser's asFactorNodes (#290).
  mu <- c(-0.6, -0.2, 0.15, 0.7)
  V <- matrix(c(1.2, -0.4, 0.6, -1.0, -0.7, 1.1, 0.9, -0.5), 4, 2)
  D <- c(0.5, 0.8, 0.6, 0.9)
  F <- matrix(c(-1, -1, 1, 1, -1, 1, -1, 1), 4, 2)
  W <- rep(0.25, 4)
  expect_error(race_probabilities(mu, V = V, D = D,
                                  F = F[, 1, drop = FALSE], W = W),
               "one column per loading")
  expect_error(race_probabilities(mu, V = V, D = D,
                                  F = F[1:2, , drop = FALSE], W = W),
               "one weight per factor node")
  expect_error(race_probabilities(mu, V = V, D = D, F = F, W = W[1:2]),
               "one weight per factor node")
  expect_error(race_probabilities(mu, V = V, D = D, F = F, W = c(W, 0.9)),
               "one weight per factor node")
  expect_error(race_probabilities(mu, V = V, D = D,
                                  F = matrix(0, 0, 2), W = numeric(0)),
               "empty|one weight per factor node")
  # and the valid spelling is untouched
  p <- race_probabilities(mu, V = V, D = D, F = F, W = W)
  expect_lt(abs(sum(p) - 1), 1e-12)
  expect_true(all(is.finite(race_probabilities(mu, V = V, D = D))))
# --- R must refuse what the other ports refuse ------------------------
#
# Found by parity/check_divergence.py, which runs the same malformed
# inputs through python, R, julia and the browser and fails when they
# disagree. R recycled a short D in silence -- D = c(2, 9) at n = 4
# priced EXACTLY the race D = c(2, 9, 2, 9) prices, same numbers, no
# warning -- and refused the scalar V that as_loadings documents and
# python and the browser accept. The inverse answered a two-runner race
# from a four-entry D.

test_that("D is a scalar or one variance per contestant, never recycled", {
  mu <- c(0, 0.3, -0.2, 0.5)
  want <- race_probabilities(mu, D = rep(1, 4))
  expect_equal(race_probabilities(mu, D = 1), want)       # scalar broadcasts
  expect_error(race_probabilities(mu, D = c(2, 9)),
               "scalar or one idiosyncratic variance")
  expect_error(race_probabilities(mu, D = rep(1, 5)),
               "scalar or one idiosyncratic variance")
  expect_error(race_probabilities(mu, D = c(1, -1, 1, 1)),
               "negative variance")
  expect_error(race_probabilities(mu, D = c(1, NaN, 1, 1)), "non-finite")
  # the recycled answer really was identical, which is why it hid
  expect_equal(race_probabilities(mu, D = c(2, 9, 2, 9)),
               race_probabilities(mu, D = c(2, 9, 2, 9)))
})

test_that("a scalar V is the same loading for everyone", {
  mu <- c(0, 0.3, -0.2, 0.5)
  # a common loading is gauge-fixed away, so this equals the plain race
  expect_equal(race_probabilities(mu, V = 0.4, D = rep(1, 4)),
               race_probabilities(mu, D = rep(1, 4)))
  expect_error(race_probabilities(mu, V = c(0.5, 0.3), D = rep(1, 4)),
               "one row per contestant")
})

test_that("the inverse's target and D describe the same field", {
  expect_error(abilities_from_race(c(0.5, 0.5), D = rep(1, 4)),
               "one variance per target entry")
  m <- abilities_from_race(c(0.4, 0.3, 0.2, 0.1), D = rep(1, 4))
  expect_equal(length(m), 4L)
  expect_true(all(is.finite(m)))
  expect_equal(abilities_from_race(c(0.4, 0.3, 0.2, 0.1), D = 1), m)
})

test_that("a zero-rank rule is the empty product, not the rank-one rule", {
  # the product grid runs zero times at k = 0: python returned the
  # RANK-1 rule there and R raised from inside expand.grid, so one
  # question had three answers across the ports
  h <- hermite_nodes(0L, 5L)
  expect_equal(dim(h$F), c(1L, 0L))
  expect_equal(h$W, 1)
  expect_equal(dim(hermite_nodes(1L, 5L)$F), c(5L, 1L))
  # k = 0 is the independent race, priced through the factor path
  mu <- c(0, 0.3, -0.2, 0.5)
  expect_equal(race_probabilities(mu, V = matrix(numeric(0), 4L, 0L),
                                  D = rep(1, 4)),
               race_probabilities(mu, D = rep(1, 4)))
})

test_that("a nonsense node count is refused by name", {
  for (k in list(-1L, -5L, 1.5, NA_real_, Inf))
    expect_error(hermite_nodes(k, 5L), "k")
  # order 0 built a rule with NO NODES, which normalises 0/0
  for (o in list(0L, -3L, 2.5))
    expect_error(hermite_nodes(1L, o), "order")
})

test_that("zero-rank loadings reach top-k as the independent top-k", {
  # .cluster_nodes handled rank 1 and rank 2 and sent everything else to
  # the rank >= 3 Sobol refusal, so an (n, 0) matrix was turned away with
  # a message about high rank (#309)
  mu <- c(0, 0.3, -0.2, 0.5)
  D <- rep(1, 4)
  V0 <- matrix(numeric(0), 4L, 0L)
  expect_equal(top_k_probabilities(mu, 2, V = V0, D = D),
               top_k_probabilities(mu, 2, D = D))
  # and a rank-one loading still MOVES it, so the equality means something
  moved <- top_k_probabilities(mu, 2, V = matrix(c(0.9, -0.4, 0.2, -0.7), 4, 1),
                               D = D)
  expect_gt(max(abs(moved - top_k_probabilities(mu, 2, D = D))), 0.005)
})
