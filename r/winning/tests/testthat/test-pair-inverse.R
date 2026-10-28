# The pair inverse keeps a tiny share whatever the label order (#412):
# c(1, 1e-16) normalizes to c(1, 1e-16) because 1 + 1e-16 == 1, and the
# closed form took qnorm(1) = Inf while the permutation was finite.
test_that("pair inverse is finite and mirrors under relabeling", {
  D <- c(0.5, 0.5)
  for (t in c(1e-12, 1e-16, 1e-20, 1e-50)) {
    a <- abilities_from_race(c(t, 1), D = D)
    b <- abilities_from_race(c(1, t), D = D)
    expect_true(all(is.finite(b)))
    expect_equal(b, rev(a), tolerance = 1e-12)
    z <- qnorm(t / (1 + t))           # the contrast at unit sd
    expect_equal(a[2] - a[1], z, tolerance = 1e-10)
  }
})

# The pair closed form is a start for a caller's factor rule, certified by
# the forward: a centred two-point law has the right two moments and a
# different pair map, and a translated rule carries a mean shift (#374).
test_that("pair inverse round-trips a non-Gaussian caller rule", {
  V <- matrix(c(0, 1), 2, 1); W <- c(.5, .5)
  for (cs in list(list(F = c(-1, 1), D = .05, m = .4),
                  list(F = c(0, 2), D = .2, m = .3))) {
    F <- matrix(cs$F, 2, 1); D <- rep(cs$D, 2)
    t <- race_probabilities(c(-cs$m, cs$m), V = V, D = D, F = F, W = W,
                            points = 1025)
    mu <- abilities_from_race(t, V = V, D = D, F = F, W = W, points = 1025)
    expect_equal(mu, c(-cs$m, cs$m), tolerance = 1e-6)
  }
})

# The core factor inverse scored only shares above max(1e-9, 1e-4/N),
# 5e-5 for a pair: a 1e-6 minority under a centred two-point law was never
# scored, and the favourite's residual certified an iterate that repriced
# it at 3.4e-19 (#727, python's _pair_solve).
test_that("the pair factor inverse scores its minority (#727)", {
  V <- matrix(c(0, 1), 2, 1); D <- c(.05, .05)
  nd <- list(F = matrix(c(-1, 1), 2, 1), W = c(.5, .5))
  for (t in c(1e-6, 1e-8)) for (target in list(c(t, 1), c(1, t))) {
    mu <- expect_silent(abilities_from_probabilities_factor(target, V, D,
                                                            nodes = nd))
    p <- win_probabilities_factor(mu, V, D, nodes = nd)
    expect_lt(max(abs(log(p / (target / sum(target))))), 1e-6)
  }
})

# With N > 2 and one share above the tail floor, the residual held no
# contrast: a self-generated [1, 1.1e-5, 5.7e-18] was certified with
# abilities 2 units off; [1, 1e-17, 3e-65] was certified with both tails
# repriced wrong. Now every share above 1e-15 (else every share) is
# scored, as python (#588).
test_that("one identified share is no certificate (#588)", {
  V <- matrix(0, 3, 1); D <- rep(1, 3)
  nd <- list(F = matrix(0, 1, 1), W = 1)
  p <- win_probabilities_factor(c(-6, 0, 6), V, D, nodes = nd, points = 1001)
  mu <- expect_silent(abilities_from_probabilities_factor(p, V, D, nodes = nd,
                                                          points = 1001))
  expect_equal(mu[2] - mu[1], 6, tolerance = 1e-4)
  q <- win_probabilities_factor(mu, V, D, nodes = nd, points = 1001)
  expect_lt(abs(log(q[2] / p[2])), 1e-5)
  p <- win_probabilities_factor(c(-12, 0, 12), V, D, nodes = nd, points = 1001)
  expect_warning(abilities_from_probabilities_factor(p, V, D, nodes = nd,
                                                     points = 1001),
                 "did not converge")
})

# Ordinary targets near the #727/#588 change: values from main before it
# (9b29ddef), so the new scoring rule moves nothing an everyday field
# reaches, and a fixed-seed battery of ordinary fields stays quiet.
test_that("ordinary factor inverses are unchanged and quiet (#727, #588)", {
  V2 <- matrix(c(0, 1), 2, 1)
  nd <- list(F = matrix(c(-1, 1), 2, 1), W = c(.5, .5))
  expect_equal(abilities_from_probabilities_factor(c(.2, .8), V2, c(.05, .05),
                                                   nodes = nd),
               c(0.54005758876225718, -0.54005758876225718), tolerance = 1e-12)
  expect_equal(abilities_from_probabilities_factor(c(.01, .99), V2, c(.5, .5)),
               c(1.6449764963002895, -1.6449764963002895), tolerance = 1e-12)
  V4 <- matrix(c(.3, -.2, .1, .4), 4, 1)
  expect_equal(abilities_from_probabilities_factor(c(.5, .3, .15, .05), V4,
                                                   c(1, .8, 1.2, .9)),
               c(-0.73544696221554973, -0.31989788152697807,
                 0.23912914904160373, 0.81621569470092403), tolerance = 1e-12)
  expect_equal(abilities_from_probabilities_factor(c(.9, .0999, 1e-4),
                                                   V4[1:3, , drop = FALSE],
                                                   c(1, 1, 1)),
               c(-2.2753196062483516, -0.35237807968419743,
                 2.6276976859325489), tolerance = 1e-12)
  set.seed(20261014)
  for (it in 1:30) {
    N <- sample(2:8, 1); r <- sample(0:2, 1)
    mu <- rnorm(N) * runif(1, .2, 2); D <- runif(N, .3, 2)
    V <- if (r == 0) matrix(0, N, 1) else matrix(rnorm(N * r) * .5, N, r)
    p <- win_probabilities_factor(mu, V, D)
    expect_silent(abilities_from_probabilities_factor(p, V, D))
  }
})
