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
