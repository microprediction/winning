test_that("node-aware window covers near-common-shock winner mass", {
  # fourteenth review's blocker: old window lost 28% of winner mass and
  # silently returned group shares 0.68/0.32 where symmetry forces 0.50
  n <- 400
  p <- block_race_probabilities(mu = rep(0, n),
                                cluster = rep(0L, n),
                                loading = c(rep(1, n/2), rep(0.9, n/2)),
                                D = rep(0.01, n), points = 1025, qa = 9)
  expect_lt(abs(sum(p[1:(n/2)]) - 0.5), 1e-4)
})

test_that("zero-strength tree traversal matches independent race", {
  # ordering the message passes by the |strength| path sum ties at zero
  # and visited children before parents (raw mass 3.0 on this tree)
  mu <- c(0.3, 0.1, 0.0, -0.1, -0.2, 0.4); mu <- mu - mean(mu)
  parent <- c(7L, 9L, 9L, 8L, 7L, 8L, 11L, 10L, 10L, 11L, 0L)
  p_t <- tree_race_probabilities(mu, cluster = 1:6, loading = rep(0, 6),
                                 D = rep(1, 6), parent = parent,
                                 strength = rep(0, 11))
  p_i <- race_probabilities(mu, D = rep(1, 6))
  expect_lt(0.5 * sum(abs(p_t - p_i)), 1e-9)
})

test_that("mass defect stops instead of normalizing", {
  expect_error(winning:::.checked_mass(c(0.5, 0.22), "test race"),
               "captured total mass")
})

test_that("tree refuses rank-r leaf loadings instead of flattening", {
  # as.numeric() on a matrix silently flattens; the port priced garbage
  set.seed(7)
  n <- 12
  mu <- rnorm(n); mu <- mu - mean(mu)
  V2 <- matrix(rnorm(2 * n) * 0.4, n, 2)
  expect_error(tree_race_probabilities(mu, rep(1:3, each = 4), V2,
                                       rep(0.6, n), c(4L, 4L, 4L, 0L),
                                       c(0, 0, 0, 0.4)),
               "rank-one")
  expect_error(block_race_jacobian(mu, rep(1:3, each = 4), V2, rep(0.6, n)),
               "rank-one")
})

test_that("mass check rejects non-finite mass", {
  expect_error(winning:::.checked_mass(c(0.5, NaN), "test race"),
               "captured total mass")
})

test_that("tree races refuse forests, cycles and out-of-range parents (#517)", {
  mu <- c(0, 1); cl <- c(1, 2); ld <- c(0, 0); D <- c(1, 1)
  # two roots: each component used to be priced alone (total mass 2)
  expect_error(tree_race_probabilities(mu, cl, ld, D, c(0, 0), c(0, 0),
                                       qa = 1),
               "exactly one root .*got 2 \\(1, 2\\)")
  expect_error(tree_race_probabilities(mu, cl, ld, D, c(NA, NA), c(0, 0),
                                       qa = 1), "exactly one root")
  expect_error(tree_race_jacobian(mu, cl, ld, D, c(0, 0), c(0, 0), qa = 1),
               "exactly one root")
  # cycle among internal nodes, with a root elsewhere
  expect_error(tree_race_probabilities(mu, cl, ld, D, c(3, 4, 4, 3, 0),
                                       c(0, 0, .1, .1, 0), qa = 1),
               "cycle: 3 -> 4 -> 3")
  expect_error(tree_race_probabilities(mu, cl, ld, D, c(3, 3, 7), c(0, 0, 0),
                                       qa = 1), "parent\\[3\\] = 7")
  expect_error(tree_race_probabilities(mu, cl, ld, D, c(2, 0), c(0, 0),
                                       qa = 1), "must be leaves")
  # a zero-strength common root is the independent race
  p <- tree_race_probabilities(mu, cl, ld, D, c(3, 3, 0), c(0, 0, 0),
                               points = 257, qa = 1)
  expect_equal(sum(p), 1, tolerance = 1e-12)
  expect_equal(p, race_probabilities(mu), tolerance = 1e-4)
})
