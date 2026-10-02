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
  ordinary <- 1  # exhaustive; 0.99998675 was the old engine's loss (#418, #362)
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

# --- exact dead heats (#418, #362, #348, #373, #369) -------------------
#
# The expected values are closed forms or brute-force enumeration of
# every joint atom outcome with a tied minimum split equally; python's
# tests/test_classic_exact_dead_heats.py checks the same fixtures.

three_atom <- function() { d <- rep(0, 17); d[8:10] <- c(.2, .3, .5); d }

test_that("a compact-support pair is priced exactly (#418)", {
  expect_lt(max(abs(state_prices_from_offsets(three_atom(), c(-1, 1)) -
                      c(0.95, 0.05))), 1e-14)
  expect_lt(max(abs(state_prices_from_offsets(three_atom(), c(-2, 2)) -
                      c(1, 0))), 1e-14)
})

test_that("random multiway ties split the claim exactly (#362)", {
  d <- c(0, 0, 0, 0, .25, .5, .25, 0, 0, 0, 0)
  expect_lt(max(abs(state_prices_from_offsets(d, rep(0, 20)) - 0.05)), 1e-14)
  want <- c(0.0918114072, 0.5380292620, 0.0029137021, 0.0918114072,
            0.0918114072, 0.0918114072, 0, 0.0918114072)
  expect_lt(max(abs(state_prices_from_offsets(d, c(0, -1, 1, 0, 0, 0, 2, 0)) -
                      want)), 1e-10)
})

test_that("fractional offsets are priced as their mixture (#348)", {
  x <- 0:82 - 41
  d <- exp(-0.5 * x^2); d <- d / sum(d)
  expect_lt(max(abs(state_prices_from_offsets(d, c(0.5, 0.5)) - 0.5)), 1e-15)
  expect_lt(abs(sum(state_prices_from_offsets(d, c(-2.5, 0.5, 3.5))) - 1), 1e-13)
})

test_that("a translated runner keeps its mass (#373)", {
  d <- rep(1 / 83, 83)
  for (k in c(0, 10, 19, 39, -19, 19.5)) {
    expect_lt(abs(state_prices_from_offsets(d, k) - 1), 1e-14)
    expect_lt(max(abs(state_prices_from_offsets(d, c(k, k)) - 0.5)), 1e-14)
  }
  ref <- state_prices_from_offsets(d, c(-5, 0, 3, 0.5))
  expect_lt(max(abs(state_prices_from_offsets(d, c(-5, 0, 3, 0.5) + 30) - ref)),
            1e-13)
})

test_that("the inverse converges to the exact forward map", {
  target <- state_prices_from_offsets(three_atom(), c(-1.5, 0, 0.7, 2))
  a <- solve_for_implied_offsets(target, three_atom(), n_iter = 10,
                                 implied_offsets_guess = rep(0, 4))
  expect_lt(max(abs(state_prices_from_offsets(three_atom(), a) - target)), 1e-6)
})

test_that("the default guess has one offset per price (#369)", {
  d <- skew_normal_density(15, 0.2, a = 1.5)
  target <- state_prices_from_offsets(d, c(6, -2))
  expect_equal(solve_for_implied_offsets(target, d),
               solve_for_implied_offsets(target, d,
                                         implied_offsets_guess = c(0, 0)))
  expect_error(solve_for_implied_offsets(target, d,
                                         implied_offsets_guess = 0:4),
               "one starting offset per price")
})

# --- the classic input contract (#339, #377) ----------------------------

test_that("raw counts are the same law as frequencies (#339)", {
  d <- skew_normal_density(50, 0.1)
  p <- state_prices_from_offsets(d, c(-3, 0.5, 2))
  expect_lt(max(abs(state_prices_from_offsets(10 * d, c(-3, 0.5, 2)) - p)), 1e-15)
})

test_that("a non-distribution is refused by forward and inverse (#339)", {
  bad <- list(c(rep(0, 7)), c(0.2, -0.1, 0.9, 0, 0, 0, 0), rep(0.25, 8),
              c(0.1, NaN, 0.9, 0, 0, 0, 0), c(0.1, Inf, 0.9, 0, 0, 0, 0),
              c(0.25, 0.5, 0.25), numeric(0))
  for (b in bad) {
    expect_error(state_prices_from_offsets(b, c(0, 0)))
    expect_error(solve_for_implied_offsets(c(0.7, 0.3), b))
  }
  expect_error(state_prices_from_offsets(c(0.25, 0.5, 0.25), c(0, 0)), "L >= 3")
})

test_that("the smallest lattice round-trips", {
  d <- c(0, 0, 0.25, 0.5, 0.25, 0, 0)
  target <- state_prices_from_offsets(d, c(0, 0))
  a <- solve_for_implied_offsets(target, d, implied_offsets_guess = c(0, 0))
  expect_true(all(is.finite(a)))
  expect_lt(max(abs(state_prices_from_offsets(d, a) - target)), 1e-12)
})

test_that("the inverse is invariant to target scale (#377)", {
  d <- skew_normal_density(500, 0.01, a = 1.5)
  p <- c(0.4, 0.3, 0.2, 0.1)
  a1 <- solve_for_implied_offsets(p, d, implied_offsets_guess = rep(0, 4))
  for (cc in c(1.1, 2, 10)) {
    ac <- solve_for_implied_offsets(cc * p, d, implied_offsets_guess = rep(0, 4))
    expect_lt(max(abs((ac - mean(ac)) - (a1 - mean(a1)))), 1e-12)
  }
  expect_error(solve_for_implied_offsets(c(0.5, -0.1, 0.6), d), "negative")
  expect_error(solve_for_implied_offsets(c(0, 0), d), "no positive mass")
  expect_error(solve_for_implied_offsets(c(0.5, NaN), d), "non-finite")
})
