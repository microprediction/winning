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

# --- state prices stay exhaustive at boundary offsets (#292) ----------
#
# state_prices_from_offsets builds the field with shifted_cdf, which
# goes through low_high and PINS any offset at or past L-2 to the
# boundary. implicit_state_prices took a fast path for exact integers
# and called integer_shift(base_cdf, k) with the RAW k, whose own clamp
# is the much wider +/-(m-1). So a runner could sit in the field at
# offset L-2 and be PAID at 60, and the prices summed to 1.8835 -- while
# an epsilon off the integer gave 0.99999, because the non-integer path
# was clamped correctly all along.

test_that("state prices sum to one at and past the boundary", {
  d <- skew_normal_density(50, 0.1)
  ordinary <- 0.9999867465
  for (a in c(48, 48.5, 49, 49.000001, 50, 60, 60.1, 100, 1000)) {
    expect_lt(abs(sum(state_prices_from_offsets(d, c(a, -a))) - ordinary),
              5e-9)
  }
})

test_that("an integer offset is continuous in its neighbourhood", {
  d <- skew_normal_density(50, 0.1)
  eps <- 1e-6
  for (k in c(49, 60, -49, -60)) {
    at <- sum(state_prices_from_offsets(d, c(k, -k)))
    lo <- sum(state_prices_from_offsets(d, c(k - eps, -(k - eps))))
    hi <- sum(state_prices_from_offsets(d, c(k + eps, -(k + eps))))
    expect_lt(abs(at - lo), 5e-9)
    expect_lt(abs(at - hi), 5e-9)
  }
})

test_that("past the clamp every offset is the same distribution", {
  d <- skew_normal_density(50, 0.1)
  a <- state_prices_from_offsets(d, c(60, -60))
  b <- state_prices_from_offsets(d, c(80, -80))
  expect_lt(max(abs(a - b)), 1e-15)
})

test_that("interior integers are untouched", {
  d <- skew_normal_density(50, 0.1)
  for (k in c(0, 1, 5, 20, 40, 47, -40)) {
    p <- state_prices_from_offsets(d, c(k, -k, k / 3))
    expect_lt(abs(sum(p) - 1), 1e-3)   # the lattice's own error, measured
    expect_true(all(p >= 0))
  }
})
