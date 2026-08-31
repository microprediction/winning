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
