
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
})


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
               "got 4 for 2 contestants")
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

test_that("the normal survival holds a 20-sd longshot (#96)", {
  p <- race_probabilities(c(0, 20, 40), D = c(1, 1, 1), points = 16385,
                          window = "span")
  expect_lt(abs(p[2] / 1.044243791881266e-45 - 1), 1e-8)   # was ~93x low
  b <- .base_normal(c(10, 20, 30))
  expect_equal(b$S, pnorm(c(10, 20, 30), lower.tail = FALSE))
})

test_that("the bulk window uses the caller's base (#106)", {
  t3 <- function(z) {
    f <- 2 / (pi * (1 + z * z)^2)
    F <- 0.5 + (atan(z) + z / (1 + z * z)) / pi
    list(S = pmax(1 - F, 1e-300), f = f, fp = -8 * z / (pi * (1 + z * z)^3))
  }
  ref <- c(0.5662010847332788, 0.3610182079892741, 0.0727807072774470)
  p <- suppressWarnings(race_probabilities(c(0, 1, 2), D = c(64, 1, 1),
                                           base = t3, points = 32001))
  expect_lt(max(abs(p - ref)), 1e-7)        # normal-survival window: 1.2e-4
  # the relaxation is announced, not silent
  expect_warning(race_probabilities(c(0, 1, 2), D = c(64, 1, 1), base = t3,
                                    points = 4001), "relaxed delta")
  # a declared span is honoured by window = "span"
  t3s <- t3; attr(t3s, "span") <- c(400, 400)
  ps <- race_probabilities(c(0, 1, 2), D = c(64, 1, 1), base = t3s,
                           points = 64001, window = "span")
  p12 <- race_probabilities(c(0, 1, 2), D = c(64, 1, 1), base = t3,
                            points = 64001, window = "span")
  expect_lt(max(abs(ps - ref)), max(abs(p12 - ref)))
})

test_that("an overflowing target sum is rescued, not refused (#326)", {
  a <- abilities_from_race(c(4e307, 2e307, 1e307, 1e307))
  b <- abilities_from_race(c(4, 2, 1, 1))
  expect_equal(a, b)
  expect_true(all(is.finite(a)))
  u <- abilities_from_race(rep(1e308, 4))
  expect_equal(u, rep(0, 4), tolerance = 1e-8)
  q <- c(1.4e308, 0.4e308, 0.2e308)
  expect_equal(abilities_from_topk(q, 1), abilities_from_topk(c(1.4, .4, .2), 1))
  expect_equal(abilities_from_rank_marginal(c(1e308, 1e308, 1e308), 1),
               rep(0, 3))
})

test_that("finite weights that overflow only in their sum are one law (#415)", {
  mu <- c(0, 0.4, 1)
  V <- matrix(c(0.7, 0, -0.4), ncol = 1)
  F <- matrix(c(-1, 1), ncol = 1)
  p1 <- race_probabilities(mu, V = V, D = rep(1, 3), F = F, W = c(1, 1),
                           points = 257L)
  pb <- race_probabilities(mu, V = V, D = rep(1, 3), F = F,
                           W = c(1e308, 1e308), points = 257L)
  expect_true(all(is.finite(pb)))
  expect_lt(max(abs(pb - p1)), 1e-15)
  # the inverse shares the normaliser
  a1 <- abilities_from_race(p1, V = V, D = rep(1, 3), F = F, W = c(1, 1))
  ab <- abilities_from_race(p1, V = V, D = rep(1, 3), F = F,
                            W = c(1e308, 1e308))
  expect_lt(max(abs(ab - a1)), 1e-10)
})

test_that("Halton nodes extend past the old 32-prime table (#402)", {
  for (r in c(32L, 33L, 40L)) {
    hw <- .halton_normal_nodes(r, 64L)
    expect_equal(dim(hw$F), c(64L, r))
    expect_true(all(is.finite(hw$F)))
  }
  # the first 32 bases are unchanged
  expect_identical(.first_primes(32L)[32], 131L)
  n <- 34L
  mu <- seq(-0.3, 0.3, length.out = n)
  V <- 0.1 * diag(n)[, seq_len(n - 1L)]
  p <- race_probabilities(mu, V = V, D = rep(1, n), points = 65L)
  expect_length(p, n)
  expect_true(all(is.finite(p)) && all(p > 0))
  expect_lt(abs(sum(p) - 1), 1e-8)
  # Python Sobol prices the same fixture in 0.0137 .. 0.0514
  expect_gt(min(p), 0.010)
  expect_lt(max(p), 0.06)
})

test_that("window is one of two names (#582)", {
  for (w in list("bulkk", "", NA, 3, TRUE))
    expect_error(race_probabilities(c(0, 1, 2), D = c(0.01, 4, 100),
                                    window = w), "window")
  expect_silent(race_probabilities(c(0, 1, 2), window = "span"))
})

test_that("the race inverse validates D and V like the forward map (#482)", {
  expect_error(abilities_from_race(c(.8, .2), D = c(-1, 2)),
               "negative variance")
  expect_error(abilities_from_race(c(.8, .2), D = c(0, 2)),
               "strictly positive")
  expect_error(abilities_from_race(c(.8, .2), D = c(NaN, 2)), "non-finite")
  expect_error(abilities_from_race(c(.5, .3, .2), D = c(1, 2)),
               "got 2 for 3")
  expect_error(abilities_from_race(c(.5, .3, .2), V = diag(2)),
               "one row per contestant")
  # a common scalar loading is gauge-invisible, as in the forward map
  expect_equal(abilities_from_race(c(.8, .2), V = 0.4, D = 1),
               abilities_from_race(c(.8, .2), D = 1))
  V <- c(.3, -.2, .5)
  expect_equal(abilities_from_race(c(.5, .3, .2), V = matrix(V, 1)),
               abilities_from_race(c(.5, .3, .2), V = V))
})
