test_that("binary case matches the closed form", {
  mu <- c(0.3, -0.4)
  V <- matrix(c(0.5, -0.2), 2, 1)
  D <- c(1.0, 1.4)
  p <- win_probabilities_factor(mu, V, D)
  s <- sqrt(sum((V[1, ] - V[2, ])^2) + sum(D))
  expect_lt(abs(p[1] - pnorm((mu[2] - mu[1]) / s)), 1e-6)
})

test_that("five-way agrees with mvtnorm", {
  skip_if_not_installed("mvtnorm")
  set.seed(7)
  n <- 5; k <- 2
  mu <- rnorm(n); V <- matrix(rnorm(n * k, 0, 0.4), n, k)
  D <- runif(n, 0.5, 1.5)
  p <- win_probabilities_factor(mu, V, D)
  Sigma <- V %*% t(V) + diag(D)
  pref <- numeric(n)
  for (i in 1:n) {
    idx <- setdiff(1:n, i)
    A <- -diag(n)[idx, , drop = FALSE]
    A[, i] <- A[, i] + 1                    # rows: X_i - X_j (min wins: < 0)
    m <- as.vector(A %*% mu)
    S <- A %*% Sigma %*% t(A)
    pref[i] <- mvtnorm::pmvnorm(
      lower = rep(-Inf, n - 1), upper = rep(0, n - 1), mean = m, sigma = S,
      algorithm = mvtnorm::GenzBretz(maxpts = 100000, abseps = 1e-7))[1]
  }
  pref <- pref / sum(pref)
  expect_lt(max(abs(p - pref)), 5e-6)
})

test_that("calibration round trip recovers shares", {
  set.seed(11)
  n <- 40; k <- 2
  mu <- rnorm(n, 0, 1.2); V <- matrix(rnorm(n * k, 0, 0.35), n, k)
  D <- runif(n, 0.5, 1.5)
  target <- win_probabilities_factor(mu, V, D)
  a <- abilities_from_probabilities_factor(target, V, D)
  back <- win_probabilities_factor(a, V, D)
  expect_lt(max(abs(back - target)), 1e-5)
})

test_that("factor inverse slopes are those of the normalised share (#371)", {
  mu <- c(.771, .170, -.474, -.467)
  D <- c(1.426, .379, .415, .363)
  V <- matrix(c(-1.946, .471, -2.087, 3.562), ncol = 1)
  nodes <- list(
    F = matrix(c(-2.02018287045609, -.958572464613819, 0,
                 .958572464613819, 2.02018287045609), ncol = 1),
    W = c(.0112574113277207, .222075922005613, .533333333333333,
          .222075922005613, .0112574113277207))
  p <- win_probabilities_factor(mu, V, D, nodes, points = 21)
  fit <- abilities_from_probabilities_factor(p, V, D, nodes, n_iter = 50,
                                             tol = 1e-8, points = 21)
  q <- win_probabilities_factor(fit, V, D, nodes, points = 21)
  expect_lt(max(abs(q - p)), 1e-7)       # was 0.21
})
