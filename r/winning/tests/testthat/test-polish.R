
# --- per-name caps are not recycled (#285's pattern) ------------------
#
# rep_len recycles in silence whenever the short length divides n, so a
# length-2 name_caps at n = 4 capped names 3 and 4 with the caps meant
# for names 1 and 2 -- and returned a perfectly well-formed constraint
# system for a problem nobody posed. Same rule as mvtnormfast's: a
# scalar broadcasts on purpose, every other wrong length is refused.

test_that("a wrong-length name_caps is refused rather than recycled", {
  for (bad in list(c(0.5, 0.3), rep(0.5, 3), rep(0.5, 5))) {
    expect_error(concentration_matrix(4, name_caps = bad),
                 "name_caps must be a scalar or one cap per name")
  }
  expect_error(concentration_matrix(4, name_caps = c(0.5, 0.3)),
               "got 2 for n = 4")
})

test_that("the documented spellings still work and agree", {
  a <- concentration_matrix(4, name_caps = 0.5)
  b <- concentration_matrix(4, name_caps = rep(0.5, 4))
  expect_equal(a$A, b$A)
  expect_equal(a$b, b$b)
  expect_equal(length(a$b), 4L)
  # NA entries are still skipped, as documented
  cc <- concentration_matrix(4, name_caps = c(0.5, NA, 0.3, NA))
  expect_equal(length(cc$b), 2L)
})

test_that("group members must be 1-based indices in 1..n (#426)", {
  for (bad in list(-1L, 0L, 1.5, NA_integer_, 5L, c(1L, -2L), integer(0)))
    expect_error(concentration_matrix(4, groups = list(list(bad, 0.5))),
                 "members must be whole numbers")
  expect_error(concentration_matrix(4, groups = list(list(1L, NA))), "cap")
  expect_error(concentration_matrix(4, groups = list(list(1L, c(.1, .2)))),
               "cap")
  expect_error(polish_race(p0 = c(0.1, 0.2, 0.3, 0.4), D = rep(1, 4),
                           groups = list(list(0L, 0.05))),
               "members must be whole numbers")
  ok <- concentration_matrix(4, groups = list(list(c(2L, 4L), 0.5)))
  expect_equal(ok$A, matrix(c(0, 1, 0, 1), nrow = 1))
  expect_equal(ok$b, 0.5)
})

test_that("max_iter bounds every phase and tol is live (#364)", {
  fx <- function(tol, max_iter)
    polish_race(mu0 = c(0, 0, 0), D = c(1, 1, 1),
                name_caps = c(0.2, NA, NA), points = 257,
                tol = tol, max_iter = max_iter)
  z <- fx(1e-9, 0)
  expect_identical(z$info$nit, 0L)
  expect_equal(z$mu, c(0, 0, 0))
  expect_false(z$info$feasible)
  expect_gt(z$info$max_violation, 0.13)
  one <- fx(1e-9, 1)
  expect_lte(max(one$info$phase_iterations), 1L)
  expect_identical(one$info$nit, sum(one$info$phase_iterations))
  full <- fx(1e-9, 60)
  expect_true(full$info$feasible)
  expect_lt(full$info$max_violation, 1e-6)
  loose <- fx(1e-2, 60)
  expect_false(identical(loose, full))
  expect_lte(loose$info$nit, full$info$nit)
  for (bad in list(0, -1, NA, Inf, c(1e-9, 1e-8)))
    expect_error(fx(bad, 10), "tol")
  for (bad in list(-1, 1.5, NA, Inf))
    expect_error(fx(1e-9, bad), "max_iter")
})

# --- race_jacobian takes the forward's window and delta (#821) ---------
#
# The forward prices on a chosen window or tail budget; the Jacobian had
# no way to take them, so it could only differentiate another lattice.

test_that("race_jacobian differentiates the forward's chosen lattice (#821)", {
  mu <- c(1.45, 0.44, 0.88, -0.30); D <- c(0.83, 1.16, 1.74, 0.34)
  # the raw lattice derivative on forward_grid(..., delta = 0.1), as the
  # python reference (_raw_and_derivative) and the browser compute it
  J01 <- matrix(c(
    -0.054290929558844736, 0.010512198572153074, 0.006387752276396653, 0.03739097871029501,
    0.010512198572153074, -0.2499025381714874, 0.03592423072716642, 0.2034661088721679,
    0.006387752276396653, 0.03592423072716642, -0.16631405225331547, 0.1240020692497524,
    0.03739097871029501, 0.2034661088721679, 0.12400206924975245, -0.36485915683221537),
    4, byrow = TRUE)
  J <- race_jacobian(mu, D = D, base = "logistic", points = 257, delta = 0.1)
  expect_lt(max(abs(J - J01)), 1e-12)
  # a span window is the span forward's derivative
  Js <- race_jacobian(mu, D = D, base = "logistic", points = 257,
                      window = "span")
  h <- 1e-5
  for (j in 1:4) {
    e <- numeric(4); e[j] <- h
    fd <- (race_probabilities(mu + e, D = D, base = "logistic", points = 257,
                              window = "span") -
           race_probabilities(mu - e, D = D, base = "logistic", points = 257,
                              window = "span")) / (2 * h)
    expect_lt(max(abs(Js[, j] - fd)), 1e-9)
  }
  # a Factor structure carries them to the same lattice
  V <- matrix(c(0.3, 0.1, -0.2, 0.4))
  expect_identical(race_jacobian(mu, structure = Factor(V, D), delta = 0.1),
                   race_jacobian(mu, V = V, D = D, delta = 0.1))
  # refused as the forward refuses them
  for (bad in list(list(delta = 0.7), list(delta = "x"),
                   list(window = "spam"))) {
    expect_error(do.call(race_probabilities, c(list(mu, D = D), bad)))
    expect_error(do.call(race_jacobian, c(list(mu, D = D), bad)),
                 "delta|window")
  }
  expect_error(race_jacobian(mu, structure = Blocks(c(1, 1, 2, 2), 0.5, D),
                             delta = 0.1), "do not support delta")
})

test_that("race_jacobian without window/delta is unchanged (#821)", {
  # the historical lattice, bit for bit: nothing given, nothing moves
  mu <- c(1.45, 0.44, 0.88, -0.30); D <- c(0.83, 1.16, 1.74, 0.34)
  J <- race_jacobian(mu, D = D, base = "logistic", points = 257)
  expect_equal(J[1, 1], -0.054290958999543587, tolerance = 1e-14)
  expect_identical(race_jacobian(mu, D = D, points = 101),
                   race_jacobian(mu, D = D, points = 101, window = NULL,
                                 delta = NULL))
})
