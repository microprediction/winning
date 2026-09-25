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
