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

test_that("hierarchical races read a scalar D as one variance each (#510)", {
  # sqrt(as.numeric(D)) kept a scalar as a one-element sd vector, so
  # sd[ord] was NA past the first contestant and the window failed
  mu <- c(-0.5, -0.1, 0.2, 0.4); mu <- mu - mean(mu)
  cl <- c(0L, 0L, 1L, 1L); L <- c(0.2, 0.3, -0.1, 0.4)
  g <- c(0.1, -0.2, 0.3, -0.1)
  expect_equal(block_race_probabilities(mu, cl, L, 1.0),
               block_race_probabilities(mu, cl, L, rep(1, 4)))
  expect_equal(nested_race_probabilities(mu, cl, L, 1.0, coupling = g,
                                         gamma = 0.5),
               nested_race_probabilities(mu, cl, L, rep(1, 4),
                                         coupling = g, gamma = 0.5))
  pa <- c(3L, 3L, 0L); st <- c(0.3, 0.3, 0.2)
  expect_equal(tree_race_probabilities(mu, cl, L, 1.0, pa, st),
               tree_race_probabilities(mu, cl, L, rep(1, 4), pa, st))
  expect_equal(block_race_jacobian(mu, cl, L, 1.0),
               block_race_jacobian(mu, cl, L, rep(1, 4)))
  p <- race_probabilities(mu, structure = Nested(cl, L, 1.0, g, 0.5))
  m <- abilities_from_race(p, structure = Nested(cl, L, 1.0, g, 0.5))
  m <- if (is.list(m)) m$mu else m
  expect_lt(max(abs(m - mu)), 1e-6)
})

test_that("rank-zero block loadings are the independent race (#508)", {
  mu <- c(-0.5, -0.1, 0.2, 0.4); D <- c(0.8, 0.9, 1.0, 1.1)
  cl <- c(0L, 0L, 1L, 1L); V0 <- matrix(numeric(0), 4, 0)
  expect_equal(block_race_probabilities(mu, cl, V0, D),
               block_race_probabilities(mu, cl, rep(0, 4), D))
  expect_equal(block_race_jacobian(mu, cl, V0, D),
               block_race_jacobian(mu, cl, rep(0, 4), D))
})

test_that("block and win/second inverses survive a sum overflow (#463, #483)", {
  q <- c(0.8, 0.6, 0.4, 0.2); cl <- c(0L, 0L, 1L, 1L)
  L <- c(0.25, -0.15, 0.35, -0.05); D <- c(0.8, 0.9, 1.0, 1.1)
  a <- abilities_from_block_race(q, cl, L, D, points = 129)
  b <- abilities_from_block_race(q * 1e308, cl, L, D, points = 129)
  a <- if (is.list(a)) a[[1]] else a; b <- if (is.list(b)) b[[1]] else b
  expect_equal(a, b, tolerance = 1e-9)
  R <- rank_probabilities(c(-0.5, -0.1, 0.2, 0.4), D = rep(1, 4))
  w <- R[, 1] / max(R[, 1]); s <- R[, 2] / max(R[, 2])
  x <- suppressWarnings(loc_scale_from_win_and_second(w, s))
  y <- suppressWarnings(loc_scale_from_win_and_second(w * 1e308, s * 1e308))
  expect_equal(x$mu, y$mu, tolerance = 1e-9)
})

test_that("a common root shock cannot move the tree race (#489)", {
  mu <- c(-0.7, -0.1, 0.2, 0.8); cl <- c(0L, 0L, 1L, 1L)
  L <- c(0.4, 0.2, 0.5, 0.1); D <- c(0.8, 1.1, 0.9, 1.2)
  b <- block_race_probabilities(mu, cl, L, D)
  for (rs in c(0, 50, 1000))
    expect_lt(max(abs(tree_race_probabilities(mu, cl, L, D, c(3L, 3L, 0L),
                                              c(0, 0, rs)) - b)), 1e-15)
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

test_that("nested coupling must have a contestant axis, never recycled (#469)", {
  mu <- c(-1.0, -0.2, 0.4, 0.8); cl <- 1:4; ld <- rep(0, 4); D <- rep(1, 4)
  for (bad in list(diag(2), c(1, 0), matrix(1, 3, 2), c(1, NA, 0, 1)))
    expect_error(nested_race_probabilities(mu, cl, ld, D, coupling = bad,
                                           points = 257, qa = 3, qf = 5),
                 "coupling")
  expect_error(nested_race_jacobian(mu, cl, ld, D, coupling = diag(2),
                                    points = 257, qa = 3, qf = 5),
               "coupling")
  G <- rbind(c(1, 0), c(0, 1), c(1, 0), c(0, 1))
  expect_equal(nested_race_probabilities(mu, cl, ld, D, coupling = t(G),
                                         points = 257, qa = 3, qf = 5),
               nested_race_probabilities(mu, cl, ld, D, coupling = G,
                                         points = 257, qa = 3, qf = 5))
})

# a scalar block loading is one loading for every contestant (#684)
test_that("scalar block loading equals the expanded vector", {
  mu <- c(-.5, -.1, .2, .4); cl <- c(0, 0, 1, 1); D <- c(.8, .9, 1, 1.1)
  expect_equal(block_race_probabilities(mu, cl, .3, D, points = 129),
               block_race_probabilities(mu, cl, rep(.3, 4), D, points = 129),
               tolerance = 0)
})

# The structured lattices were laid at the caller's origin: at a common
# offset of 1e13 the block, nested and tree races raised a mass defect
# (zero mass at 1e15) and their Jacobians came back zero. The contrasts
# below are exact in binary64 (#477, python's blocks._home).
test_that("a common ability offset reaches no structured lattice (#477)", {
  base <- c(-1, 0, 1); cl <- c(1, 2, 3); ld <- c(.3, .2, .4); D <- c(1, 1, 1)
  cp <- c(.3, -.1, -.2); par <- c(4, 4, 5, 5, 0); st <- c(.5, .5, .3, .3, 0)
  f <- list(
    function(x) block_race_probabilities(x, cl, ld, D),
    function(x) nested_race_probabilities(x, cl, ld, D, coupling = cp),
    function(x) tree_race_probabilities(x, cl, ld, D, par, st),
    function(x) block_race_jacobian(x, cl, ld, D),
    function(x) nested_race_jacobian(x, cl, ld, D, coupling = cp),
    function(x) tree_race_jacobian(x, cl, ld, D, par, st),
    function(x) win_probabilities_factor(x, matrix(cp, 3, 1), D))
  for (c in c(1e12, 1e13, 1e15)) {
    x <- base + c
    expect_identical(x - x[2], base)
    for (g in f) expect_equal(g(x), g(base), tolerance = 1e-14)
  }
})

# Only gamma g is identified: coupling 1e308 g with gamma 0.8e-308
# overflowed g %*% node before the tiny gamma was applied (forward and
# Jacobian failed on a NaN), and the inverse's variance helper formed
# 0 * Inf from 1e307 (#722).
test_that("nested reciprocal coupling scaling is the same race (#722)", {
  mu <- c(-.4, -.1, .2, .5, .8); cl <- c(0, 0, 1, 1, 2)
  ld <- c(.4, .2, .3, .5, .1); D <- c(.7, .8, .9, 1, 1.1)
  g <- c(-.4, .1, .4, -.2, .1)
  mk <- function(c) Nested(cl, ld, D, g * c, 0.8 / c)
  t0 <- race_probabilities(mu, structure = mk(1))
  J0 <- nested_race_jacobian(mu, cl, ld, D, g, 0.8)
  a0 <- abilities_from_race(t0, structure = mk(1))
  for (c in c(1e307, 1e308)) {
    expect_equal(race_probabilities(mu, structure = mk(c)), t0,
                 tolerance = 1e-14)
    expect_equal(nested_race_jacobian(mu, cl, ld, D, g * c, 0.8 / c), J0,
                 tolerance = 1e-14)
    expect_equal(abilities_from_race(t0, structure = mk(c)), a0,
                 tolerance = 1e-10)
  }
})

# Ordinary inputs near the #477/#722 changes keep main's values (9b29ddef).
test_that("ordinary nested races are unchanged (#477, #722)", {
  mu <- c(-.4, -.1, .2, .5, .8); cl <- c(0, 0, 1, 1, 2)
  ld <- c(.4, .2, .3, .5, .1); D <- c(.7, .8, .9, 1, 1.1)
  g <- c(-.4, .1, .4, -.2, .1)
  ref <- c(0.38432227045253309, 0.23696573454430217, 0.18328372050474345,
           0.12254165848578566, 0.072886616012635769)
  expect_equal(nested_race_probabilities(mu, cl, ld, D, coupling = g,
                                         gamma = .8), ref, tolerance = 1e-12)
  expect_equal(nested_race_probabilities(mu + 3.7, cl, ld, D, coupling = g,
                                         gamma = .8), ref, tolerance = 1e-12)
})

test_that("tree scalar loading equals its expansion (#780)", {
  # .scalar_loading(.3) kept length one and v[ord] filled NA for the rest
  mu <- c(-.5, -.1, .2, .4); cl <- c(1L, 1L, 2L, 2L)
  D <- c(.8, .9, 1, 1.1); pa <- c(3L, 3L, 0L)
  for (st in list(c(0, 0, .4), c(0, 0, 0))) {
    expect_identical(
      tree_race_probabilities(mu, cl, .3, D, pa, st, points = 129),
      tree_race_probabilities(mu, cl, rep(.3, 4), D, pa, st, points = 129))
    expect_identical(
      tree_race_jacobian(mu, cl, .3, D, pa, st, points = 129),
      tree_race_jacobian(mu, cl, rep(.3, 4), D, pa, st, points = 129))
  }
  p <- tree_race_probabilities(mu, cl, rep(.3, 4), D, pa, c(0, 0, .4),
                               points = 129)
  expect_lt(max(abs(p - c(0.42684000, 0.25205727, 0.17948344, 0.14161929))),
            1e-6)
})

test_that("rank-two block loadings are canonical per cluster (#787)", {
  mu <- c(2.43, .27, -.01, -.19, 1.07, -1.03); cl <- c(0, 0, 0, 1, 1, 1)
  D <- c(1, 1.39, 1.48, .85, .94, .61)
  V <- matrix(c(.69, -.65, .29, -.95, -.69, -1.15, -.04, -1.28, .61, -.27,
                -.20, 2.02), ncol = 2, byrow = TRUE)
  h <- sqrt(.5); Q <- matrix(c(h, h, -h, h), 2); Vr <- V %*% Q
  expect_lt(max(abs(Vr %*% t(Vr) - V %*% t(V))), 1e-15)
  p <- block_race_probabilities(mu, cl, V, D)
  expect_lt(max(abs(p - block_race_probabilities(mu, cl, Vr, D))), 1e-14)
  expect_lt(max(abs(p - race_probabilities(mu, structure = Blocks(cl, Vr, D)))),
            1e-14)
  g <- rep(.3, 6)
  expect_lt(max(abs(nested_race_probabilities(mu, cl, V, D, coupling = g) -
                    nested_race_probabilities(mu, cl, Vr, D, coupling = g))),
            1e-14)
  # per-cluster rotation, a column sign and a column swap
  Vc <- V; Vc[1:3, ] <- V[1:3, ] %*% Q
  for (W in list(Vc, V %*% diag(c(-1, 1)), V[, 2:1]))
    expect_lt(max(abs(p - block_race_probabilities(mu, cl, W, D))), 1e-14)
  # python's canonical price
  expect_lt(max(abs(p - c(0.0016660074987572178, 0.08641098200675351,
                          0.15703265532332839, 0.26876786125311775,
                          0.02195242706689778, 0.4641700668511454))), 1e-13)
  # rank one untouched; zero-padded and rank-deficient rank two are rank one
  v <- V[, 1]
  r1 <- block_race_probabilities(mu, cl, v, D)
  expect_lt(max(abs(r1 - c(0.00244822624473019, 0.1180086074271331,
                           0.19963224309434435, 0.14866389613894831,
                           0.03235943719199677, 0.4988875899028474))), 1e-14)
  expect_lt(max(abs(r1 - block_race_probabilities(mu, cl, cbind(v, 0), D))),
            1e-15)
  expect_lt(max(abs(block_race_probabilities(mu, cl, sqrt(5) * v, D) -
                    block_race_probabilities(mu, cl, cbind(v, 2 * v), D))),
            1e-14)
})
