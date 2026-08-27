# The new API: structures, kernels, Jacobians, inversion, polish.
# Cross-language parity is pinned separately by parity/check.R at the
# repo root; these are self-consistency invariants.

set.seed(7)
n <- 8
mu <- rnorm(n)
D <- 0.5 + runif(n)
cl <- rep(1:4, each = 2)
ld <- 0.2 + 0.3 * runif(n)

test_that("structure grammars agree on their containments", {
  p_ind <- race_probabilities(mu, structure = Independent(D))
  p_blk0 <- race_probabilities(mu, structure = Blocks(cl, rep(0, n), D))
  expect_lt(max(abs(p_ind - p_blk0)), 1e-10)
  p_blk <- race_probabilities(mu, structure = Blocks(cl, ld, D))
  p_nst0 <- race_probabilities(mu, structure = Nested(cl, ld, D, ld, gamma = 0))
  expect_lt(max(abs(p_blk - p_nst0)), 1e-14)
  # blocks = tree of depth 1 (root strength 0)
  p_tree <- race_probabilities(mu,
    structure = Tree(cl, ld, D, parent = c(5, 5, 5, 5, 0), strength = rep(0, 5)))
  expect_lt(max(abs(p_blk - p_tree)), 1e-9)
})

test_that("block jacobian matches finite differences and has zero row sums", {
  J <- block_race_jacobian(mu, cl, ld, D, points = 257)
  expect_lt(max(abs(rowSums(J))), 1e-10)
  h <- 1e-5
  for (j in c(1, 5)) {
    e <- numeric(n); e[j] <- h
    fd <- (block_race_probabilities(mu + e, cl, ld, D, points = 257) -
             block_race_probabilities(mu - e, cl, ld, D, points = 257)) / (2 * h)
    expect_lt(max(abs(J[, j] - fd)), 1e-5)
  }
})

test_that("block inversion round trips", {
  p <- c(0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04)
  out <- abilities_from_block_race(p, cl, ld, D)
  expect_lt(out$residual, 1e-8)
  back <- block_race_probabilities(out$mu, cl, ld, D)
  expect_lt(max(abs(back - p)), 1e-8)
})

test_that("gumbel base with D = pi^2/6 is exactly softmax", {
  m <- c(-0.5, 0, 0.4, 1)
  p <- race_probabilities(m, D = rep(pi^2 / 6, 4), base = "gumbel",
                          points = 1001)
  expect_lt(max(abs(p - exp(-m) / sum(exp(-m)))), 1e-6)
})

test_that("polish_race enforces caps and stays a race", {
  p0 <- c(0.4, 0.25, 0.2, 0.15)
  out <- polish_race(p0 = p0, D = rep(1, 4), name_caps = 0.3)
  expect_lte(max(out$p), 0.3 + 1e-6)
  expect_lt(abs(sum(out$p) - 1), 1e-9)
  # unconstrained names move as little as possible: order preserved
  expect_true(all(diff(order(out$p)) == diff(order(p0))))
})

test_that("bulk window matches span window", {
  V <- matrix(rnorm(n * 2) * 0.3, n, 2)
  pb <- race_probabilities(mu, V = V, D = D, points = 257, window = "bulk")
  ps <- race_probabilities(mu, V = V, D = D, points = 1001, window = "span")
  expect_lt(max(abs(pb - ps)), 1e-7)
})
