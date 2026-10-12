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

.first_primes <- function(d) {
  # generated, not tabulated: a fixed table put a hard cliff in the public
  # cov= route, which stopped at 31 runners and then threw (#190)
  if (d < 1L) return(integer(0))
  out <- integer(d)
  out[1] <- 2L
  k <- 1L
  cand <- 3L
  while (k < d) {
    isp <- TRUE
    lim <- floor(sqrt(cand))
    for (q in out[seq_len(k)]) {
      if (q > lim) break
      if (cand %% q == 0L) { isp <- FALSE; break }
    }
    if (isp) { k <- k + 1L; out[k] <- cand }
    cand <- cand + 2L
  }
  out
}

.halton_unit <- function(d, n, skip = 20L) {
  primes <- .first_primes(d)
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
# k with x 2^(2k) of order one (x > 0 finite), else 0.
.pow2_half_exponent <- function(x) {
  if (!(is.finite(x) && x > 0)) return(0)
  -(floor(log2(x)) + 1) %/% 2
}

# Lower Cholesky factor of a contrast covariance, AS GIVEN whenever it has
# one (python: winning.methods.native._contrast_cholesky). An absolute
# 1e-12 ridge was model noise in the utility unit (#101, #409), and the
# relative ridge that replaced it, added always, still inflated a valid
# small pivot (near-twins with variance 2^-43 priced [0.5, 0.167, 0.333]
# for the exact [0.5, 0.079, 0.421]), and its 1e-300 floor was an absolute
# ridge below that scale (max error 0.159 at scale 2^-525). The matrix is
# brought to unit scale by an exact power of two, factorised unmodified,
# given a ridge of 1e-12 of its unit mean variance only if that fails, and
# the factor scaled back exactly. An exactly zero contrast (a tie, Dirac
# scope) keeps the old floor.
.contrast_chol <- function(C) {
  m <- nrow(C)
  tr <- sum(diag(C)) / m
  if (!(is.finite(tr) && tr > 0))
    return(t(chol(C + diag(1e-12 * max(tr, 1e-300), m))))
  k <- .pow2_half_exponent(tr)
  Cs <- C * 2^k * 2^k
  U <- tryCatch(chol(Cs), error = function(e) NULL)
  if (is.null(U)) U <- chol(Cs + diag(1e-12 * sum(diag(Cs)) / m, m))
  t(U) * 2^-k
}

.ghk_one <- function(mu, Sigma, i, u, want_slope = FALSE) {
  n <- length(mu)
  others <- seq_len(n)[-i]
  a <- mu[others] - mu[i]
  M <- matrix(0, n - 1L, n)
  M[cbind(seq_len(n - 1L), others)] <- 1
  M[, i] <- M[, i] - 1
  # Sigma at unit scale by an exact power of two first, so a race in tiny
  # units is differenced in normal, not subnormal, arithmetic (#409)
  ks <- .pow2_half_exponent(mean(diag(Sigma)))
  Cc <- M %*% (Sigma * 2^ks * 2^ks) %*% t(M)
  L <- .contrast_chol(Cc) * 2^-ks
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
#'
#' Halton needs one prime per contestant less one, and those primes are
#' generated rather than tabulated: a 30-entry table made the public cov=
#' route throw at 32 runners, on inputs that had returned a factor-fit
#' answer before (#190). High-dimensional projections are the known
#' weakness of the family, so the agreement was measured rather than
#' assumed -- against python on exactly-rank-3 correlations, max|R - py| is
#' 2.7e-4 at n=16, 5.6e-4 at n=32 and 3.6e-4 at n=48 (0.1s, 0.3s, 0.8s).
#' It degrades gently and stays far inside the 4.6e-3 that pricing the
#' degraded fit costs, so there is no size at which falling back would be
#' the better answer.
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
