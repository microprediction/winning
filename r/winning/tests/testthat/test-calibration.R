# Tests for r/winning's dividend and price conversions.
# winning/research/pricing.py::StatePricer is the specification.

# --- dividends that are not ordinary positive numbers (#242) ----------
#
# Only a MISSING quote -- NA or NaN -- becomes nan_value. This divided by
# the dividend unconditionally, so a dividend of 0 gave Inf and then NaN
# after normalising, and a NEGATIVE dividend came back as a negative
# "probability". The expected values are python's
# StatePricer.prices_from_dividends, which the browser now matches too.

test_that("a worthless dividend prices at zero, not NaN or negative", {
  want <- c(2 / 3, 1 / 3, 0)
  for (d in list(c(2, 4, Inf), c(2, 4, 0), c(2, 4, -5), c(2, 4, -Inf))) {
    p <- prices_from_dividends(d)
    expect_true(all(is.finite(p)), info = paste(d, collapse = ","))
    expect_true(all(p >= 0), info = paste(d, collapse = ","))
    expect_lt(max(abs(p - want)), 1e-12)
  }
})

test_that("an all-infinite book is zeros, not 0/0", {
  p <- prices_from_dividends(c(Inf, Inf))
  expect_identical(as.numeric(p), c(0, 0))
})

test_that("a missing quote still takes nan_value", {
  p <- prices_from_dividends(c(2, 4, NA))
  expect_lt(abs(p[3] - 1 / 2000 / (1 / 2 + 1 / 4 + 1 / 2000)), 1e-12)
  expect_lt(abs(sum(p) - 1), 1e-12)
})

test_that("ordinary books are unchanged", {
  p <- prices_from_dividends(c(2, 4, 8, 8))
  expect_lt(max(abs(p - c(0.5, 0.25, 0.125, 0.125) / 1.0)), 1e-12)
})
