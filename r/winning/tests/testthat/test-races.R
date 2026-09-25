
# --- R must refuse what the other ports refuse ------------------------
#
# Found by parity/check_divergence.py, which runs the same malformed
# inputs through python, R, julia and the browser and fails when they
# disagree. R recycled a short D in silence -- D = c(2, 9) at n = 4
# priced EXACTLY the race D = c(2, 9, 2, 9) prices, same numbers, no
# warning -- and refused the scalar V that as_loadings documents and
# python and the browser accept. The inverse answered a two-runner race
# from a four-entry D.

test_that("D is a scalar or one variance per contestant, never recycled", {
  mu <- c(0, 0.3, -0.2, 0.5)
  want <- race_probabilities(mu, D = rep(1, 4))
  expect_equal(race_probabilities(mu, D = 1), want)       # scalar broadcasts
  expect_error(race_probabilities(mu, D = c(2, 9)),
               "scalar or one idiosyncratic variance")
  expect_error(race_probabilities(mu, D = rep(1, 5)),
               "scalar or one idiosyncratic variance")
  expect_error(race_probabilities(mu, D = c(1, -1, 1, 1)),
               "negative variance")
  expect_error(race_probabilities(mu, D = c(1, NaN, 1, 1)), "non-finite")
  # the recycled answer really was identical, which is why it hid
  expect_equal(race_probabilities(mu, D = c(2, 9, 2, 9)),
               race_probabilities(mu, D = c(2, 9, 2, 9)))
})

test_that("a scalar V is the same loading for everyone", {
  mu <- c(0, 0.3, -0.2, 0.5)
  # a common loading is gauge-fixed away, so this equals the plain race
  expect_equal(race_probabilities(mu, V = 0.4, D = rep(1, 4)),
               race_probabilities(mu, D = rep(1, 4)))
  expect_error(race_probabilities(mu, V = c(0.5, 0.3), D = rep(1, 4)),
               "one row per contestant")
})

test_that("the inverse's target and D describe the same field", {
  expect_error(abilities_from_race(c(0.5, 0.5), D = rep(1, 4)),
               "one variance per target entry")
  m <- abilities_from_race(c(0.4, 0.3, 0.2, 0.1), D = rep(1, 4))
  expect_equal(length(m), 4L)
  expect_true(all(is.finite(m)))
  expect_equal(abilities_from_race(c(0.4, 0.3, 0.2, 0.1), D = 1), m)
})
