
# --- W and c*W are the same factor law (#281's defect, R's copy) ------
#
# The forward divides its accumulated shares by their total, so it was
# invariant either way. race_jacobian does not, so J(W) and J(10W)
# differed by 1.473 -- a Newton step scaled by the spelling of the law
# rather than the law. Same defect as #281 in the standalone javascript
# module and #290 in the browser tree; python has never had it, because
# _setup puts W through winning.shapes.as_weights.

test_that("the forward and the jacobian are invariant to W -> cW", {
  mu <- c(-0.6, -0.2, 0.15, 0.7)
  V <- matrix(c(1.2, -0.4, 0.6, -1.0, -0.7, 1.1, 0.9, -0.5), 4, 2)
  D <- c(0.5, 0.8, 0.6, 0.9)
  F <- matrix(c(-1, -1, 1, 1, -1, 1, -1, 1), 4, 2)
  W <- rep(0.25, 4)
  p0 <- race_probabilities(mu, V = V, D = D, F = F, W = W)
  j0 <- race_jacobian(mu, V = V, D = D, F = F, W = W)
  for (cc in c(1e-6, 0.1, 10, 1e6)) {
    expect_lt(max(abs(race_probabilities(mu, V = V, D = D, F = F,
                                         W = W * cc) - p0)), 1e-15)
    expect_lt(max(abs(race_jacobian(mu, V = V, D = D, F = F,
                                    W = W * cc) - j0)), 1e-15)
  }
  # an unnormalised spelling of the SAME law, not a rescaling
  expect_lt(max(abs(race_probabilities(mu, V = V, D = D, F = F,
                                       W = c(1, 1, 1, 1)) - p0)), 1e-15)
})

test_that("a W that is not a law is refused", {
  mu <- c(-0.6, -0.2, 0.15, 0.7)
  V <- matrix(c(1.2, -0.4, 0.6, -1.0, -0.7, 1.1, 0.9, -0.5), 4, 2)
  D <- c(0.5, 0.8, 0.6, 0.9)
  F <- matrix(c(-1, -1, 1, 1, -1, 1, -1, 1), 4, 2)
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = c(0.5, -0.5, 0.5, 0.5)),
               "non-negative")
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = rep(0, 4)),
               "positive total")
  expect_error(race_probabilities(mu, V = V, D = D, F = F,
                                  W = c(0.25, NA, 0.25, 0.25)),
               "positive total|finite")
})
