# Round trips referee the R port directly (the parity harness referees
# it against python; this keeps R CMD check self-contained).

test_that("top-k round trip recovers mean-zero locations", {
  mu <- c(-1.1, -0.4, 0.0, 0.3, 0.7, 0.5)
  mu <- mu - mean(mu)
  for (k in c(1, 2, 4)) {
    q <- top_k_probabilities(mu, k)
    back <- abilities_from_topk(q, k)
    expect_lt(max(abs(back - mu)), 5e-5)
  }
})

test_that("rank marginals are doubly stochastic and cumulative", {
  mu <- c(-0.8, -0.2, 0.1, 0.9)
  P <- rank_probabilities(mu)
  expect_lt(max(abs(rowSums(P) - 1)), 1e-8)
  expect_lt(max(abs(colSums(P) - 1)), 1e-8)
  q2 <- top_k_probabilities(mu, 2)
  expect_lt(max(abs(rowSums(P[, 1:2]) - q2)), 1e-6)
})

test_that("loc/scale pair identifies both parameters up to gauge", {
  mu <- c(-0.7, -0.1, 0.2, 0.6, 0.0)
  sd <- c(0.9, 1.15, 1.0, 0.95, 1.05)
  cc <- exp(mean(log(sd)))
  q1 <- top_k_probabilities(mu, 1, D = sd^2)
  q2 <- top_k_probabilities(mu, 2, D = sd^2)
  fit <- loc_scale_from_topk_pair(q1, 1, q2, 2)
  expect_lt(max(abs(fit$mu - (mu - mean(mu)) / cc)), 1e-3)
  expect_lt(max(abs(fit$sd - sd / cc)), 1e-3)
})

test_that("jacobians satisfy the gauge identities", {
  mu <- c(-0.5, 0.0, 0.2, 0.6)
  D <- c(1.2, 0.9, 1.0, 1.1)
  J <- top_k_jacobians(mu, 2, D = D)
  expect_lt(max(abs(rowSums(J$Jmu))), 1e-10)
  euler <- J$Jmu %*% mu + J$Jsigma %*% sqrt(D)
  expect_lt(max(abs(euler)), 1e-7)
})

test_that("the returned rank matrix is what gets checked", {
  # Row normalisation makes rows exact and MOVES the columns, so a matrix
  # that passed the raw check could fail the stated identity afterwards
  # with nothing looking (#203). Columns are not forced: the same field at
  # 2001 points has a column error of 3.5e-9, so the quadrature is what is
  # wrong, and raising says so.
  mu <- c(0.852479112355, 1.342705472309, -4.927823648082)
  sd <- c(0.635892136598, 0.194587863958, 14.30720080209)
  # #224 refines the grid to the NARROWEST runner, so this field resolves
  # itself now instead of being rejected. The guard remains as the
  # backstop for what refinement cannot reach, past the 8193 cap.
  P513 <- rank_probabilities(mu, D = sd^2, points = 513)
  expect_lt(max(abs(rowSums(P513) - 1)), 1e-6)
  expect_lt(max(abs(colSums(P513) - 1)), 1e-6)
  expect_error(suppressWarnings(
    rank_probabilities(c(0, 1, -1), D = c(5e-5, 1, 3)^2, points = 513)),
    "defective")
  P <- rank_probabilities(mu, D = sd^2, points = 2001)
  expect_lt(max(abs(rowSums(P) - 1)), 5e-3)
  expect_lt(max(abs(colSums(P) - 1)), 5e-3)
  q <- top_k_probabilities(mu, 2, D = sd^2, points = 2001)
  expect_lt(max(abs(rowSums(P[, 1:2]) - q)), 1e-6)
})

test_that("a gross raw defect is caught before normalisation", {
  # The guard #221 added, on a field the refinement CANNOT rescue: sds
  # spanning 8636x, so the point count it asks for is past the 8193 cap
  # and the raw matrix really is defective. Same field as python's
  # MU_UNRESOLVABLE / SD_UNRESOLVABLE.
  mu <- c(-5.629882, -10.123372, -4.986196, 0.330608)
  sd <- c(12.4970445, 0.00144703, 3.06481372, 0.00690731)
  expect_error(rank_probabilities(mu, D = sd^2, points = 513),
               "before normalisation")
})

test_that("the field that used to be defective now resolves", {
  # When #221 was written this field (sds spanning 232x) had a raw matrix
  # off by 0.249 in a row, and dividing by those same row sums left a
  # column defect of 0.0047 -- inside the tolerance, so the
  # post-normalisation check alone returned a false success on it.
  #
  # #224 then refined the lattice to resolve the NARROWEST runner: the
  # window [-210.45, 110.81] at 513 requested points expands to 6845, and
  # the field resolves at every requested resolution. Pinning it as a
  # failure would pin a bug that is fixed. The python reference records
  # the same thing (#243).
  mu <- c(1.7802347516623231, 7.71387298475329,
          -8.864593234372917, -13.723882426584884)
  sd <- c(0.09388844913109345, 0.2459800972322556,
          1.0565710909345742, 21.775572080525823)
  P <- rank_probabilities(mu, D = sd^2, points = 513)
  expect_true(all(is.finite(P)))
  expect_lt(max(abs(rowSums(P) - 1)), 1e-12)
  expect_lt(max(abs(colSums(P) - 1)), 1e-6)
  P2 <- rank_probabilities(mu, D = sd^2, points = 2001)
  expect_lt(max(abs(P - P2)), 1e-6)
})

# --- a top-k depth is a count, so it is an integer --------------------
#
# The guard truncated with as.integer(k) and then range-checked the
# TRUNCATED value, so k=0, k=n and k>n were all refused and only a
# non-integer slipped through, silently floored: top_k_probabilities(
# mu, 1.5) returned the top-1 curve with mass 1, and 2.5 the top-2 one.
# The caller asked for a curve that does not exist and got a different
# one, and the mass they can check is 1 or 2, not the 1.5 they asked
# about. All three ports shared this identically.

test_that("whole depths are accepted and fractional ones refused", {
  mu <- c(-0.4, 0.1, 0.2, 0.5); D <- c(0.7, 0.8, 0.9, 1.0)
  for (k in c(1, 2, 3)) {
    p <- top_k_probabilities(mu, k, D = D)
    expect_lt(abs(sum(p) - k), 1e-9)
  }
  for (k in c(1.5, 2.5, 0.5, 2.0001)) {
    expect_error(top_k_probabilities(mu, k, D = D), "whole number of places")
  }
  for (k in c(0, 4, 5)) {
    expect_error(top_k_probabilities(mu, k, D = D), "\\[1, n-1\\]")
  }
})

test_that("the pair door checks both depths", {
  mu <- c(-0.4, 0.1, 0.2, 0.5); D <- c(0.7, 0.8, 0.9, 1.0)
  q1 <- top_k_probabilities(mu, 1, D = D)
  q2 <- top_k_probabilities(mu, 2, D = D)
  expect_silent(loc_scale_from_topk_pair(q1, 1, q2, 2))
  expect_error(loc_scale_from_topk_pair(q1, 1.5, q2, 2),
               "k1 must be a whole number")
  expect_error(loc_scale_from_topk_pair(q1, 1, q2, 2.5),
               "k2 must be a whole number")
  expect_error(loc_scale_from_topk_pair(q1, 0, q2, 2), "\\[1, n-1\\]")
  expect_error(loc_scale_from_topk_pair(q1, 1, q2, 1), "k1 == k2")
})

test_that("an iteration budget of zero performs no updates (#441)", {
  q <- c(0.70, 0.20, 0.10)
  f0 <- abilities_from_topk(q, k = 1, points = 513, n_iter = 0,
                            return_info = TRUE)
  f1 <- abilities_from_topk(q, k = 1, points = 513, n_iter = 1,
                            return_info = TRUE)
  f2 <- abilities_from_topk(q, k = 1, points = 513, n_iter = 2,
                            return_info = TRUE)
  logt <- log(q)
  expect_equal(f0$mu, -(logt - mean(logt)) / 2)   # the initializer
  expect_identical(f0$iterations, 0L)
  expect_identical(f1$iterations, 1L)
  expect_identical(f2$iterations, 2L)
  expect_false(isTRUE(all.equal(f0$mu, f2$mu)))
  # the residual describes the RETURNED mu
  for (f in list(f0, f1, f2)) {
    qh <- top_k_probabilities(f$mu, 1, points = 513)
    r <- max(abs(qlogis(qh) - qlogis(q / sum(q))))
    expect_equal(f$max_logit_residual, r, tolerance = 1e-10)
  }
  for (bad in list(-1, 1.5, NA, Inf, c(1, 2), "a")) {
    expect_error(abilities_from_topk(q, 1, n_iter = bad), "n_iter")
    expect_error(abilities_from_rank_marginal(q, 1, n_iter = bad), "n_iter")
  }
  mu <- c(-0.4, 0.1, 0.2, 0.5)
  q1 <- top_k_probabilities(mu, 1); q2 <- top_k_probabilities(mu, 2)
  expect_error(loc_scale_from_topk_pair(q1, 1, q2, 2, n_iter = -1), "n_iter")
  z <- loc_scale_from_topk_pair(q1, 1, q2, 2, n_iter = 0, return_info = TRUE)
  expect_identical(z$iterations, 0L)
  z <- abilities_from_rank_marginal(q, 1, n_iter = 0, return_info = TRUE)
  expect_identical(z$iterations, 0L)
  expect_equal(z$mu, rep(0, 3))
})

test_that("top-k and rank doors refuse wrong-length D (#319)", {
  mu <- c(-0.3, 0.1, 0.0, 0.2)
  for (D in list(c(1, 4), c(1, 1, 1, 1, 1), c(1, 0, 1, 1), c(1, -1, 1, 1),
                 c(1, NA, 1, 1), c(1, Inf, 1, 1))) {
    expect_error(top_k_probabilities(mu, 2, D = D), "D")
    expect_error(top_k_jacobians(mu, 2, D = D), "D")
    expect_error(rank_probabilities(mu, D = D), "D")
    q <- top_k_probabilities(mu, 2)
    expect_error(abilities_from_topk(q, 2, D = D), "D")
    expect_error(abilities_from_rank_marginal(q / 2, 1, D = D), "D")
    expect_error(race_probabilities(mu, D = D), "D")
  }
  q1 <- top_k_probabilities(mu, 1); q2 <- top_k_probabilities(mu, 2)
  expect_error(loc_scale_from_topk_pair(q1, 1, q2, 2, D0 = c(1, 4)), "D0")
  # scalar broadcasts, length n accepted
  expect_equal(top_k_probabilities(mu, 2, D = 2),
               top_k_probabilities(mu, 2, D = rep(2, 4)))
})

test_that("top-k doors refuse V without one row per runner (#343)", {
  mu <- c(-.8, -.2, .4, 1); D <- rep(1, 4)
  V_bad <- matrix(c(-2, 2), ncol = 1)
  expect_error(top_k_probabilities(mu, 2, V = V_bad, D = D), "one row per")
  expect_error(top_k_jacobians(mu, 2, V = V_bad, D = D), "one row per")
  q <- top_k_probabilities(mu, 2, D = D)
  expect_error(abilities_from_topk(q, 2, V = V_bad, D = D), "one row per")
  expect_error(top_k_probabilities(mu, 2, V = matrix(1, 3, 1)), "one row per")
  expect_error(top_k_probabilities(mu, 2, V = c(1, 2, 3)), "one row per")
  v <- c(-1, 0.5, 0.2, 0.3)
  a <- top_k_probabilities(mu, 2, V = v)
  expect_equal(top_k_probabilities(mu, 2, V = matrix(v, ncol = 1)), a)
  expect_equal(top_k_probabilities(mu, 2, V = matrix(v, nrow = 1)), a)
  # scalar common loading is the independent race after gauge fixing
  expect_equal(top_k_probabilities(mu, 2, V = 0.7),
               top_k_probabilities(mu, 2), tolerance = 1e-6)
})

test_that("rank_probabilities prices a factor-correlated race (#202)", {
  # R had no V/qa here, so the correlated call was an unused-argument
  # error while top_k_probabilities priced the same model
  mu <- c(-0.4, -0.1, 0.2, 0.5)
  D <- rep(0.2, 4)
  V <- matrix(c(-2, -0.5, 0.5, 2), ncol = 1)
  R <- rank_probabilities(mu, V = V, D = D, qa = 15)
  q1 <- top_k_probabilities(mu, 1, V = V, D = D, qa = 15)
  q2 <- top_k_probabilities(mu, 2, V = V, D = D, qa = 15)
  expect_lt(max(abs(R[, 1] - q1)), 1e-6)
  expect_lt(max(abs(rowSums(R[, 1:2]) - q2)), 1e-6)
  # Python reference, same fixture
  expect_lt(max(abs(R[, 1] - c(0.53003131, 0.08960483, 0.049621,
                               0.33074286))), 1e-6)
  # and it differs from the independent answer (0.331 vs 0.031)
  expect_gt(abs(R[4, 1] - rank_probabilities(mu, D = D)[4, 1]), 0.2)
  # rank two
  V2 <- cbind(V, c(0.3, -0.2, 0.4, -0.5))
  R2 <- rank_probabilities(mu, V = V2, D = D, qa = 9)
  expect_lt(max(abs(R2[, 1] - top_k_probabilities(mu, 1, V = V2, D = D,
                                                  qa = 9))), 1e-6)
})
