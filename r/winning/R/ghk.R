# GHK for the dense-covariance race, mirroring winning.factor.races._race_dense
# and winning.methods.native._ghk_prob.
#
# The grammar fit degenerates on some covariances -- full rank with D on its
# floor, where the conditional race is a near-step the factor nodes cannot
# resolve, and the residual checks stay silent. The python package routes
# that case here instead of pricing the fit. This port used to have no GHK
# and priced the fit anyway, 4.6e-3 away.
#
# Two properties are not optional, both learned the hard way in python:
#
#   * GHK conditions the runners sequentially, so its answer depends on the
#     order they are listed in. A fixed point set made one 8-runner field
#     price 8.8e-3 apart under two labelings. The runners are sorted into a
#     canonical order before the estimate and the answer is mapped back,
#     which makes the route exactly permutation-equivariant -- and that
#     order is also the accurate one.
#   * Everything stays in log space. A runner far behind underflows to zero
#     probability, and the inverse Newton-steps on log residuals: a warm
#     start three sd out returned P = 0 and the iteration had nowhere to go.
#
# The uniforms are a dependency-free Halton sequence where python uses
# scrambled Sobol, so the two agree to the node families' own difference
# rather than exactly; .ghk_race() documents the measured figure.

.halton_unit <- function(d, n, skip = 20L) {
  primes <- c(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113)
  if (d > length(primes))
    stop("GHK needs more Halton dimensions than this table holds")
  out <- matrix(0, n, d)
  for (j in seq_len(d)) {
    b <- primes[j]
    for (i in seq_len(n)) {
      k <- i + skip
      f <- 1 / b
      h <- 0
      while (k > 0) {
        h <- h + f * (k %% b)
        k <- k %/% b
        f <- f / b
      }
      out[i, j] <- h
    }
  }
  pmin(pmax(out, 1e-12), 1 - 1e-12)
}

# log P(U_i is the maximum) for U ~ N(mu, Sigma), by GHK sequential
# conditioning. With want_slope, also log dP_i/dmu_i, accumulated in the
# same pass as the score of the conditional product: sum_t phi(b_t) /
# (Phi(b_t) L_tt), holding the truncated draws fixed. The exact derivative
# of the estimator also carries the draws' dependence on mu through the
# truncation, which this omits, so it is an own-slope for a preconditioner
# and not a gradient for an optimiser (measured 5-25% above central
# differences, which the adaptive damping absorbs).
.ghk_one <- function(mu, Sigma, i, u, want_slope = FALSE) {
  n <- length(mu)
  others <- seq_len(n)[-i]
  a <- mu[others] - mu[i]
  M <- matrix(0, n - 1L, n)
  M[cbind(seq_len(n - 1L), others)] <- 1
  M[, i] <- M[, i] - 1
  Cc <- M %*% Sigma %*% t(M)
  L <- t(chol(Cc + diag(1e-12, n - 1L)))
  R <- nrow(u)
  z <- matrix(0, R, n - 1L)
  logprob <- numeric(R)
  score <- numeric(R)
  for (t in seq_len(n - 1L)) {
    if (t == 1L) {
      bb <- rep(-a[1L] / L[1L, 1L], R)
    } else {
      bb <- as.vector(-a[t] - z[, seq_len(t - 1L), drop = FALSE] %*%
                        L[t, seq_len(t - 1L)]) / L[t, t]
    }
    lF <- stats::pnorm(bb, log.p = TRUE)
    logprob <- logprob + lF
    if (want_slope)
      score <- score + exp(stats::dnorm(bb, log = TRUE) - lF) / L[t, t]
    z[, t] <- stats::qnorm(pmin(pmax(u[, t] * exp(lF), 1e-300), 1 - 1e-16))
  }
  lse <- function(v) { m <- max(v); m + log(sum(exp(v - m))) }
  out <- list(logP = lse(logprob) - log(R))
  if (want_slope)
    out$logS <- lse(logprob + log(pmax(score, 1e-300))) - log(R)
  out
}

#' Dense-covariance race by GHK (internal)
#'
#' Min-wins, like the rest of the package, so mu is negated for the
#' max-wins GHK. Returns log probabilities, and with slopes also
#' d log p_i / d mu_i (negative).
#'
#' Measured against the python package's scrambled-Sobol route on its own
#' n=8 fixtures: agreement is the node families' difference, which the
#' tests pin. Halton here keeps the package dependency-free, as elsewhere.
#' @keywords internal
.ghk_race <- function(mu, C, budget = 4096L, want_slopes = FALSE) {
  C <- as.matrix(C)
  n <- nrow(C)
  mu <- as.numeric(mu)
  # canonical order: by ability, ties by covariance row sum
  ord <- order(mu, rowSums(C))
  Cs <- C[ord, ord, drop = FALSE]
  ms <- -mu[ord]                                    # max-wins
  u <- .halton_unit(n - 1L, as.integer(budget))
  res <- lapply(seq_len(n), function(i)
    .ghk_one(ms, Cs, i, u, want_slope = want_slopes))
  lp <- vapply(res, function(r) r$logP, numeric(1))
  shift <- max(lp)
  w <- exp(lp - shift)
  tot <- sum(w)
  logp_sorted <- lp - shift - log(tot)
  logp <- numeric(n)
  logp[ord] <- logp_sorted
  if (!want_slopes) return(list(logp = logp, p = exp(logp)))
  ls <- vapply(res, function(r) r$logS, numeric(1))
  dlogp_sorted <- -exp(ls - lp)                     # min-wins: negate
  dlogp <- numeric(n)
  dlogp[ord] <- dlogp_sorted
  list(logp = logp, p = exp(logp), dlogp = dlogp)
}
