
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
