
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
