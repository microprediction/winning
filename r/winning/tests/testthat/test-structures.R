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

# the shared fixture tests/golden/linkage_leaf_variance.json (#430): leaf
# variances are 2 h^2 EXACTLY -- no absolute 1e-10 floor -- and a merge at
# height 0 is refused; python and the browser read the same file
test_that("near-duplicate leaves keep their cophenetic variance (#430)", {
  path <- file.path("..", "..", "..", "..", "tests", "golden",
                    "linkage_leaf_variance.json")
  skip_if_not(file.exists(path), "shared fixture only in the repo checkout")
  skip_if_not_installed("jsonlite")
  fx <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  for (case in fx$cases) {
    Z <- do.call(rbind, lapply(case$Z, unlist))
    tr <- tree_from_linkage(Z)
    D <- unlist(case$D)
    expect_equal(tr$D, D, tolerance = fx$rtol, info = case$name)
    n <- length(D)
    st <- unlist(case$strength)
    expect_equal(tr$strength[-(1:n)], st[-(1:n)], tolerance = fx$rtol,
                 info = case$name)
  }
  Z0 <- do.call(rbind, lapply(fx$refuse_zero_height, unlist))
  expect_error(tree_from_linkage(Z0), "height 0")
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

# linkage topology and heights are checked at the constructor (#328, #459)
test_that("tree_from_linkage refuses malformed linkages", {
  bad <- list(rbind(c(0, 1, -0.5, 2), c(2, 3, 0.8, 3)),
              rbind(c(0, 1, NaN, 2), c(2, 3, 0.8, 3)),
              rbind(c(3, 0, 0.5, 2), c(1, 2, 0.8, 3)),
              rbind(c(0, 0, 0.5, 2), c(1, 2, 0.8, 3)),
              rbind(c(0, 1, 0.5, 2), c(0, 2, 0.8, 3)),
              rbind(c(0, 1.5, 0.5, 2), c(2, 3, 0.8, 3)))
  for (Z in bad) expect_error(tree_from_linkage(Z), "linkage row")
  expect_equal(tree_from_linkage(rbind(c(0, 1, 0.5, 2), c(2, 3, 0.8, 3)))$D,
               c(0.5, 0.5, 1))
})

# an explicit leaf map makes Tree pricing independent of label spelling (#372)
test_that("Tree leaves map makes relabeling invariant", {
  mu <- c(-0.5, -0.1, 0.2, 0.4); D <- c(0.8, 0.9, 1.0, 1.1)
  L <- c(0.2, 0.3, -0.1, 0.4); pa <- c(4, 4, 5, 5, 0); st <- c(0, 0, 0, 0.6, 0.4)
  a <- tree_race_probabilities(mu, c(0, 0, 1, 2), L, D, pa, st)
  b <- tree_race_probabilities(mu, c(30, 30, 10, 20), L, D, pa, st,
                               leaves = c(30, 10, 20))
  expect_equal(a, b, tolerance = 0)
  T <- Tree(c(30, 30, 10, 20), L, D, pa, st, leaves = c(30, 10, 20))
  expect_equal(race_probabilities(mu, structure = T),
               race_probabilities(mu, structure = Tree(c(0, 0, 1, 2), L, D, pa, st)),
               tolerance = 0)
  expect_error(tree_race_probabilities(mu, c(30, 30, 10, 20), L, D, pa, st,
                                       leaves = c(30, 10)), "does not map")
})

test_that("every grammar prices a pair in closed form (#86)", {
  ex <- pnorm(-20 / sqrt(2))
  for (s in list(Blocks(c(0, 1), c(0, 0), c(1, 1)),
                 Nested(c(0, 1), c(0, 0), c(1, 1), c(0, 0), 0.5),
                 Tree(c(0, 1), c(0, 0), c(1, 1), c(3, 3, 0), c(0, 0, 0))))
    expect_lt(abs(race_probabilities(c(0, 20), structure = s)[2] / ex - 1), 1e-12)
  expect_lt(abs(race_probabilities(c(0, 20), D = c(1, 1))[2] / ex - 1), 1e-12)
  s <- Tree(c(0, 1), c(.8, -.5), c(1, 2), c(3, 4, 4, 0), c(0, 0, .9, 1.5))
  expect_lt(abs(race_probabilities(c(0, 1.3), structure = s)[1] -
                  pnorm(1.3 / sqrt(.64 + .81 + .25 + 3))), 1e-14)
})
