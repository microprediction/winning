test_that("fit_covariance recovers in-grammar truths", {
  set.seed(1)
  n <- 30
  V <- matrix(rnorm(n * 2), n, 2) * 0.5
  D <- 0.5 + runif(n)
  C <- V %*% t(V) + diag(D)
  mu <- sort(rnorm(n))
  fit <- fit_covariance(C)
  p1 <- race_probabilities(mu, cov = C)
  p0 <- race_probabilities(mu, V = V, D = D, F = fit$F[, 1:2] * 0 + 
                             .halton_normal_nodes(2, 2048)$F,
                           W = .halton_normal_nodes(2, 2048)$W)
  expect_lt(0.5 * sum(abs(p1 - p0)), 2e-3)
})

test_that("cov= inversion round trip", {
  set.seed(2)
  n <- 20
  V <- matrix(rnorm(n * 2), n, 2) * 0.5
  C <- V %*% t(V) + diag(0.5 + runif(n))
  mu0 <- sort(rnorm(n)); mu0 <- mu0 - mean(mu0)
  p <- race_probabilities(mu0, cov = C)
  mu_hat <- abilities_from_race(p, cov = C)
  expect_lt(max(abs(mu_hat - mu0)), 1e-4)
})

test_that("a degraded cov= fit is announced where it cannot be routed", {
  # The normal race with no slopes is routed to GHK (see below). Every
  # other call needs the factor form -- slopes for the inverse's
  # preconditioner, a non-normal base -- and prices the degraded fit, so
  # it says so. It used to say nothing at all on any path, which is the
  # one failure mode that looks like an answer.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.05 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.6, 0.6, length.out = n)
  expect_warning(race_probabilities(mu, cov = C, return_slopes = TRUE),
                 "degraded")
  expect_warning(abilities_from_race(c(.4, .2, .1, .1, .07, .06, .04, .03),
                                     cov = C, base = "gumbel"), "degraded")
  fit <- fit_covariance(C)
  expect_true(fit$degraded)
  expect_true(fit$rank <= n)          # #180: the eigen arm used to ask for
                                      # more columns than there are
})

test_that("a healthy cov= fit is priced without a warning", {
  set.seed(7)
  n <- 20
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.3 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.8, 0.8, length.out = n)
  expect_silent(p <- race_probabilities(mu, cov = C, points = 257))
  expect_equal(sum(p), 1, tolerance = 1e-9)
  expect_false(fit_covariance(C)$degraded)
})

test_that("a degraded cov= fit is routed to GHK, forward and inverse", {
  # python routes this case rather than pricing the fit; so does this port
  # now, to the same canonical runner order and in log space throughout.
  # Routing only the forward would make the two front doors describe
  # different races, which is python's #164.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  d <- 0.05 + runif(n)
  S <- V %*% t(V) + diag(d)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- seq(-0.6, 0.6, length.out = n)
  expect_silent(p <- race_probabilities(mu, cov = C))   # routed: no warning
  expect_equal(sum(p), 1, tolerance = 1e-12)
  mu_back <- abilities_from_race(p, cov = C)
  expect_equal(race_probabilities(mu_back, cov = C), p, tolerance = 1e-7)
  expect_equal(mean(mu_back), 0, tolerance = 1e-9)
})

test_that("GHK is permutation-equivariant and its slopes have the right sign", {
  set.seed(11)
  n <- 6
  A <- matrix(rnorm(n * n), n, n)
  S <- A %*% t(A) + n * diag(n)
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  mu <- rnorm(n)
  g <- .ghk_race(mu, C, want_slopes = TRUE)
  expect_equal(sum(g$p), 1, tolerance = 1e-12)
  expect_true(all(g$dlogp < 0))       # min-wins: a higher mu loses mass
  perm <- c(4, 1, 6, 2, 5, 3)
  gp <- .ghk_race(mu[perm], C[perm, perm])$p
  back <- numeric(n); back[perm] <- gp
  expect_equal(back, g$p, tolerance = 1e-12)
})


test_that("GHK has no dimension cliff", {
  # a 30-prime table made the public cov= route throw at 32 runners, on
  # inputs that returned a factor-fit answer before (#190)
  expect_equal(.first_primes(5), c(2, 3, 5, 7, 11))
  expect_equal(length(.first_primes(64)), 64)
  for (n in c(31, 32, 40)) {
    g <- .ghk_race(rep(0, n), diag(n))
    expect_equal(sum(g$p), 1, tolerance = 1e-10)
    expect_equal(g$p, rep(1 / n, n), tolerance = 1e-6)   # exchangeable
  }
})

test_that("degradation is judged against the clamp close_fit actually uses", {
  # the test compared D against 1e-6 * diag while close_fit clamped at
  # 1e-3 * mean(diag), so a clamp-bound fit counted zero bound entries and
  # was priced instead of routed, restoring the divergence #188 closed.
  # The clamp is now RELATIVE to each runner's variance (#407); for a
  # correlation matrix it is the same 1e-3 as before, and the contract
  # pinned here -- bound is counted against the floor actually used -- is
  # unchanged.
  set.seed(3)
  n <- 8
  V <- matrix(rnorm(n * 3), n, 3)
  S <- V %*% t(V) + diag(0.05 + runif(n))
  s <- sqrt(diag(S))
  C <- S / outer(s, s)
  fit <- fit_covariance(C)
  expect_true(fit$bound > 0)                    # was 0 before the fix
  expect_equal(fit$clamp, rep(1e-3 * mean(diag(C)), n))
  expect_true(fit$degraded)
  expect_true(is.finite(fit$contrast_residual))
})

test_that("a two-runner covariance fits instead of throwing", {
  # The closing solve is against P o P = a I + b 11' with a = 1 - 2/n, and
  # at n = 2 that a is exactly zero, so the matrix is rank one and solve()
  # threw for EVERY 2x2 input, public cov= calls included (#181). Python
  # has had the two-runner branch all along.
  for (rho in c(0, 0.6, 0.95, -0.4)) {
    C <- matrix(c(1, rho, rho, 1), 2, 2)
    fit <- fit_covariance(C, k = 1L, m = 1L, nodes = 32L)
    expect_equal(length(fit$D), 2L)
    expect_true(all(fit$D > 0))
    p <- race_probabilities(c(0, 0.5), cov = C)
    expect_equal(sum(p), 1, tolerance = 1e-12)
    expect_true(p[1] > p[2])            # min-wins: the lower mu wins more
  }
  expect_silent(fit_covariance(diag(2), k = 1L, m = 1L, nodes = 32L))
})

# --- a one-runner covariance (#273) -----------------------------------
#
# race_probabilities(mu) already returned 1 for a single runner.
# cov = matrix(v, 1, 1) crashed instead: .factor_model_projected is asked
# for min(k, n - 1) = 0 factors and hands back a 0 x 0 matrix, which then
# fails to multiply. Python divided by zero at the same field, since its
# inner solver divides by (1 - 2/n) + n/n^2 = 0 at n = 1.
#
# There is nothing for a factor to correlate with one runner, so the
# whole variance is idiosyncratic and the fit is exact at rank zero.

test_that("a one-runner race with a covariance is certain", {
  p <- race_probabilities(2.0, cov = matrix(4.0, 1, 1))
  expect_length(p, 1L)
  expect_lt(abs(p[1] - 1), 1e-12)
})

test_that("it agrees with the plain one-runner race", {
  expect_lt(abs(race_probabilities(2.0) -
                race_probabilities(2.0, cov = matrix(4.0, 1, 1))), 1e-12)
})

test_that("the one-runner fit reports rank zero and no residual", {
  f <- fit_covariance(matrix(4.0, 1, 1))
  expect_identical(f$rank, 0L)
  expect_identical(f$degraded, FALSE)
  expect_lt(abs(f$D[1] - 4.0), 1e-9)
  expect_lt(abs(sum(f$W) - 1), 1e-12)
})

test_that("larger fields are untouched", {
  for (n in c(2L, 3L, 5L)) {
    C <- diag(n) + 0.2 * (matrix(1, n, n) - diag(n))
    p <- race_probabilities(seq(0, 1, length.out = n), cov = C)
    expect_length(p, n)
    expect_lt(abs(sum(p) - 1), 1e-9)
    expect_true(all(p > 0))
  }
})

# --- the validation the one-runner branch sits below (#277) ----------
#
# The early return keys off nrow alone, so it has to come AFTER the
# shape and covariance checks: a 1 x 2 matrix has nrow 1, and would
# otherwise be answered from C[1, 1] with the second column dropped.
# R had no validation here at all -- a negative variance was clamped to
# the floor and NA/Inf propagated into the fit -- while python refused
# each one. Same contract, same messages, both ports.

test_that("a non-square cov is refused rather than read by nrow", {
  expect_error(fit_covariance(matrix(c(1, 2), 1, 2)), "must be square")
  expect_error(fit_covariance(matrix(c(1, 2), 2, 1)), "must be square")
  # the message names the shape it got
  expect_error(fit_covariance(matrix(c(1, 2), 1, 2)), "1 x 2")
})

test_that("a non-finite one-runner cov is refused", {
  expect_error(fit_covariance(matrix(NA_real_, 1, 1)), "NaN or inf")
  expect_error(fit_covariance(matrix(Inf, 1, 1)), "NaN or inf")
  expect_error(fit_covariance(matrix(-Inf, 1, 1)), "NaN or inf")
  expect_error(fit_covariance(matrix(NaN, 1, 1)), "NaN or inf")
})

test_that("a negative one-runner variance is refused, not clamped", {
  # it used to come back as D = 1e-12, i.e. invalid data dressed up as
  # a certain race
  expect_error(fit_covariance(matrix(-1, 1, 1)), "positive semidefinite")
})

test_that("a zero one-runner variance is valid and lands on the floor", {
  f <- fit_covariance(matrix(0, 1, 1))
  expect_equal(as.numeric(f$D), 1e-12)     # python's value too
  expect_equal(as.numeric(f$W), 1)
  expect_equal(f$rank, 0L)
})

test_that("an asymmetric cov is refused at every size", {
  expect_error(fit_covariance(matrix(c(1, 0.9, 0.1, 1), 2, 2)),
               "not symmetric")
})

test_that("a valid one-runner cov still fits", {
  f <- fit_covariance(matrix(4, 1, 1))
  expect_equal(as.numeric(f$D), 4)
  expect_equal(as.numeric(f$V), 0)
  expect_equal(f$rank, 0L)
})

# --- symmetrising must not overflow what the check just passed (#279)
#
# The finiteness check runs before the symmetrisation, and
# 0.5 * (C + t(C)) doubles before it halves, so a finite variance near
# the double ceiling passed the check and then became Inf. The PSD
# tolerance had the same shape: sum(diag(C)) overflows before the
# division by n, and a tolerance of -Inf makes the comparison FALSE
# for every eigenvalue, so the check accepted a matrix that is not a
# covariance at all.

test_that("a huge finite one-runner variance survives symmetrising", {
  for (v in c(1e307, 1e308, 1.5e308, .Machine$double.xmax)) {
    f <- fit_covariance(matrix(v, 1, 1))
    expect_true(is.finite(f$D))
    expect_equal(as.numeric(f$D), v)
  }
})

test_that("an overflowing trace does not disable the PSD check", {
  C <- matrix(c(1e308, 1.5e308, 1.5e308, 1e308), 2, 2)
  expect_lt(min(eigen(C, symmetric = TRUE, only.values = TRUE)$values), 0)
  expect_error(fit_covariance(C), "positive semidefinite")
})

test_that("a large two-runner cov still fits", {
  # 1e308 overflows further into the fit, which is separate work
  for (v in c(1e200, 1e300)) {
    f <- fit_covariance(diag(c(v, v)))
    expect_true(all(is.finite(f$D)))
  }
})

test_that("the GHK ridge is relative: a race in tiny units is the same race (#101, #409)", {
  # binary: one truncation step, so exact. Min-wins, so runner 1 (mu 0)
  # beats runner 2 (mu c) with probability Phi(1/sqrt 2) at every scale;
  # the absolute 1e-12 ridge gave 0.5040 at c = 1e-8
  for (cc in c(1, 1e-6, 1e-8)) {
    g <- .ghk_race(c(0, cc), diag(cc^2, 2), budget = 64L)
    expect_lt(abs(g$p[1] - pnorm(1 / sqrt(2))), 1e-10)
  }
})

test_that("near-singular contrasts survive the closing floor (#407)", {
  C <- diag(c(1e-8, 1e-8, 1))
  f <- fit_covariance(C)
  S <- f$V %*% t(f$V) + diag(f$D)
  expect_equal(f$D[1:2], c(1e-8, 1e-8), tolerance = 1e-6)
  expect_lt(abs((S[1, 1] + S[2, 2] - 2 * S[1, 2]) / 2e-8 - 1), 1e-6)
  expect_lt(f$contrast_residual, 1e-6)
  expect_false(f$degraded)
  exact <- pnorm(-0.001 / sqrt(2e-8))
  p <- suppressWarnings(race_probabilities(c(0, 0.001, 10), cov = C,
                                           return_slopes = TRUE,
                                           points = 65537)$p)
  expect_lt(abs(p[2] / exact - 1), 0.05)       # was 0.48 (a coin flip)
  # heterogeneous scales: each floor is relative to its own variance, so
  # every pairwise contrast variance is held (D itself is not identified
  # at n = 3, only the choice-relevant contrasts are)
  C2 <- diag(c(1e-6, 1e-6, 1, 100, 100))
  f2 <- fit_covariance(C2)
  expect_lt(f2$contrast_residual, 1e-5)
  expect_true(all(f2$clamp == 1e-3 * pmax(diag(C2), 1e-6 * mean(diag(C2)))))
})

test_that("the closing solve is the lower-bounded least squares (#407)", {
  # an active floor: the unconstrained solve goes negative on runner 1
  set.seed(21)
  n <- 6
  V <- matrix(rnorm(n * 2), n, 2)
  C <- V %*% t(V) + diag(c(1e-9, rep(1, n - 1)))
  f <- fit_covariance(C, k = 1L, m = 0L, blocks = 1L)
  P <- diag(n) - 1 / n
  obj <- function(D) sum((P %*% (C - f$V %*% t(f$V) - diag(D)) %*% P)^2)
  fl <- f$clamp
  expect_true(all(f$D >= fl - 1e-18))
  # KKT: no feasible coordinate move lowers the objective
  for (i in seq_len(n)) for (h in c(-1, 1) * 1e-5) {
    D2 <- f$D; D2[i] <- max(D2[i] + h, fl[i])
    expect_gte(obj(D2), obj(f$D) - 1e-12)
  }
})

test_that("an exact rank-2-plus-diagonal matrix is returned undistorted (#427)", {
  V0 <- matrix(c(
     0.075438132656, -0.079262917975,  0.384253590266,  0.062940070292,
    -0.321401623897,  0.216957032946,  0.782400027078,  0.568248577878,
    -0.422241141484, -0.759252882628, -0.373964677522,  0.024795587608,
    -1.395018464783, -0.131274998360, -0.747546568352, -0.439360412822,
    -0.326555389714, -0.189780093821,  0.246978321824,  0.625508021666,
    -0.077120797766,  0.819878082330, -0.399116804092,  0.210906042056),
    ncol = 2L, byrow = TRUE)
  D0 <- c(1.701258166559, 0.585268443588, 9.872334718900, 9.155259899265,
          2.350087156121, 1.999486851270, 2.381735122844, 0.599574077550,
          0.186291487317, 2.773171194144, 1.123850763479, 0.417333983859)
  C <- tcrossprod(V0) + diag(D0)
  f <- fit_covariance(C, k = 2L, m = 0L, blocks = 1L, nodes = 32L)
  Cfit <- tcrossprod(f$V) + diag(f$D)
  P <- diag(12) - 1 / 12
  expect_lt(max(abs(P %*% (C - Cfit) %*% P)), 1e-6)     # was 0.295
  expect_lt(f$contrast_residual, 1e-6)                    # was 0.159
  expect_false(f$degraded)
  dv <- function(S, i, j) S[i, i] + S[j, j] - 2 * S[i, j]
  expect_lt(abs(pnorm(1.471517470795 / sqrt(dv(Cfit, 11, 12))) -
                pnorm(1.471517470795 / sqrt(dv(C, 11, 12)))), 1e-6)
})
