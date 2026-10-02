# tree_from_linkage promises the cophenetic matrix, or says no (#133).
# Python's Tree.from_linkage refuses an inversion; this is its parity.

# three runners: {0,1} merge at 0.5, then {2, {0,1}} at 0.3 -- a merge
# below its own child, which no nested variance decomposition represents
Z_inverted <- matrix(c(0, 1, 0.5, 2,
                       2, 3, 0.3, 3), 2, 4, byrow = TRUE)

test_that("an inverted linkage is refused, not clipped", {
  expect_error(tree_from_linkage(Z_inverted),
               "not monotonic: node 3 merges 0.32 BELOW its parent, and 1 node")
})

test_that("tree_from_hclust refuses the same inversion", {
  hc <- structure(list(merge = matrix(c(-1L, -3L, -2L, 1L), 2),
                       height = c(0.5, 0.3), order = c(3L, 1L, 2L)),
                  class = "hclust")
  expect_error(tree_from_hclust(hc), "not monotonic")
})

test_that("a monotonic linkage is unaffected", {
  Z <- Z_inverted; Z[2, 3] <- 0.6
  tr <- tree_from_linkage(Z)
  expect_equal(length(tr$parent), 5L)
  hc <- hclust(dist(matrix(c(0, 0.1, 0.5, 0.2, 0.3, 0.9), 6)) / 10,
               method = "average")
  expect_equal(length(tree_from_hclust(hc)$parent), 11L)
})

test_that("an empty linkage is the one-runner tree (#382)", {
  tr <- tree_from_linkage(matrix(numeric(0), nrow = 0, ncol = 4))
  expect_equal(tr$D, 1)
  expect_length(tr$cluster, 1L)
  expect_length(tr$parent, 1L)
  expect_length(tr$strength, 1L)
  expect_equal(race_probabilities(0, structure = tr), 1)
  # a two-leaf tree is unchanged by the guard
  tr2 <- tree_from_linkage(matrix(c(0, 1, 0.5, 2), nrow = 1))
  expect_equal(tr2$D, c(0.5, 0.5))
})
