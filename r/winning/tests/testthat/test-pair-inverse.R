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
