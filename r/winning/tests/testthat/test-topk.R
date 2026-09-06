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
