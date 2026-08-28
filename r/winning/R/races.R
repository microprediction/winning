# The general race: one API, distributions and correlation as parameters.
# Base-R port of winning/factor/races.py (the python reference is the
# spec; parity/vectors.json pins the two together).
#
# Min-wins convention throughout. A base is a function z -> list(S, f, fp)
# giving survival, density and density derivative of a mean-zero,
# unit-variance law.

.EULER <- 0.5772156649015329

.base_normal <- function(z) {
  S <- pmax(1 - stats::pnorm(z), 1e-300)
  f <- exp(-0.5 * z^2) / sqrt(2 * pi)
  list(S = S, f = f, fp = -z * f)
}

.base_gumbel <- function(z) {
  cc <- pi / sqrt(6)
  u <- pmin(z * cc - .EULER, 30)
  eu <- exp(u)
  S <- pmax(exp(-eu), 1e-300)
  f <- cc * eu * S
  list(S = S, f = f, fp = cc * cc * eu * S * (1 - eu))
}

.BASES <- list(normal = .base_normal, gumbel = .base_gumbel)
.SPANS <- list(normal = c(8, 8), gumbel = c(22, 8))

.hermite1 <- function(order) {
  off <- sqrt(seq_len(order - 1))
  J <- matrix(0, order, order)
  J[cbind(seq_len(order - 1), seq_len(order - 1) + 1)] <- off
  J[cbind(seq_len(order - 1) + 1, seq_len(order - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  idx <- order(e$values)
  w <- e$vectors[1, idx]^2
  list(nodes = e$values[idx], weights = w / sum(w))
}

# Dependency-free Halton sequence mapped through qnorm: equal-weight
# nodes for E over N(0, I_r). Used when the sharpness escalation calls
# for a low-discrepancy family (see .race_setup); adequate for r <= 4.
.halton_normal_nodes <- function(r, n) {
  primes <- c(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
              47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131)[seq_len(r)]
  H <- vapply(primes, function(b) {
    idx <- seq_len(n) + 20L          # drop the first few, standard hygiene
    h <- numeric(n)
    f <- 1 / b
    i <- idx
    while (any(i > 0)) {
      h <- h + f * (i %% b)
      i <- i %/% b
      f <- f / b
    }
    h
  }, numeric(n))
  F <- qnorm(pmin(pmax(H, 1e-12), 1 - 1e-12))
  list(F = matrix(F, ncol = r), W = rep(1 / n, n))
}

.race_setup <- function(mu, V, D, F, W, base) {
  mu <- as.numeric(mu)
  n <- length(mu)
  if (is.null(D)) D <- rep(1, n)
  D <- as.numeric(D)
  if (is.null(V)) {
    V <- matrix(0, n, 1)
    F <- matrix(0, 1, 1)
    W <- 1
  } else {
    V <- as.matrix(V)
    if (is.null(F) || is.null(W)) {
      # adaptive order matching the python reference: sharp conditional
      # races (small D relative to loadings) need more factor nodes
      sharp <- max(sqrt(rowSums(V^2)) / sqrt(pmax(D, 1e-300)))
      r <- ncol(V)
      if (r >= 2 && sharp > 3.0) {
        # matching the python reference: past this sharpness the factor
        # integrand is a near-step and Gauss-Hermite converges slowly at
        # any order; escalate the FAMILY to a low-discrepancy rule.
        # Python uses scrambled Sobol; here dependency-free Halton.
        hw <- .halton_normal_nodes(r, 2^13)
        F <- hw$F
        W <- hw$W
      } else {
        cap <- if (r == 1) 201 else if (r == 2) 41 else 15
        Q <- as.integer(min(max(ceiling(8 * sharp), 15), cap))
        hw <- hermite_nodes(ncol(V), order = Q)
        F <- hw$F
        W <- hw$W
      }
    }
  }
  fn <- if (is.function(base)) base else .BASES[[base]]
  span <- if (is.function(base)) c(12, 12) else {
    s <- .SPANS[[base]]
    if (is.null(s)) c(12, 12) else s
  }
  list(mu = mu, V = V, D = D, F = as.matrix(F), W = as.numeric(W),
       fn = fn, left = span[1], right = span[2])
}

# Lattice over the WINNER distribution's bulk, not the ability span --
# port of races._bulk_window (bisection on a conservative winner-cdf
# envelope, 2 sd base-agnostic pad).
.bulk_window <- function(M_all, sd, points, delta) {
  mu_lo <- apply(M_all, 2, min)
  mu_hi <- apply(M_all, 2, max)
  s <- sd
  G <- function(x) {
    logS <- log(pmax(1 - stats::pnorm((x - mu_lo) / s), 1e-300))
    1 - exp(sum(logS))
  }
  H <- function(x) {
    logS <- log(pmax(1 - stats::pnorm((x - mu_hi) / s), 1e-300))
    1 - exp(sum(logS))
  }
  lo0 <- min(mu_lo) - 9 * max(s)
  hi0 <- max(mu_hi) + 9 * max(s)
  a <- lo0; b <- hi0
  for (i in 1:80) {
    m <- 0.5 * (a + b)
    if (G(m) < delta) a <- m else b <- m
  }
  xlo <- a
  a <- xlo; b <- hi0
  for (i in 1:80) {
    m <- 0.5 * (a + b)
    if (H(m) < 1 - delta) a <- m else b <- m
  }
  pad <- 2 * max(s)
  seq(xlo - pad, b + pad, length.out = points)
}

#' Win probabilities of the general race, all N in one field pass
#'
#' @param mu numeric abilities (min-wins: lower is better)
#' @param V optional N x k loading matrix
#' @param D idiosyncratic variances (default 1)
#' @param F,W optional factor nodes and weights
#' @param base "normal", "gumbel", or a function z -> list(S, f, fp)
#' @param points lattice size (default 257)
#' @param return_slopes also return d p_raw_i / d mu_i (inversion
#'   preconditioner), normalized as p is
#' @param structure optional covariance grammar (see
#'   \code{\link{Independent}}); overrides V/D
#' @param window "bulk" (winner-bulk lattice, default) or "span"
#' @param delta omitted winner mass bound for the bulk window
#' @param qa,qf quadrature orders for structure dispatch
#' @param nodes deprecated alias for list(F, W)
#' @return probabilities summing to one, or list(p, slopes)
#' @export
race_probabilities <- function(mu, V = NULL, D = NULL, F = NULL, W = NULL,
                               base = "normal", points = 257,
                               return_slopes = FALSE, structure = NULL,
                               window = "bulk", delta = 1e-12,
                               qa = 9, qf = 15, nodes = NULL, cov = NULL) {
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    V <- fit$V; D <- fit$D; F <- fit$F; W <- fit$W
  }
  if (!is.null(structure)) {
    return(.dispatch_probabilities(mu, structure, base = base,
                                   points = points, qa = qa, qf = qf,
                                   return_slopes = return_slopes))
  }
  if (!is.null(nodes)) { F <- nodes$F; W <- nodes$W }
  st <- .race_setup(mu, V, D, F, W, base)
  n <- length(st$mu)
  sd <- sqrt(st$D)
  Q <- nrow(st$F)
  M_all <- matrix(st$mu, Q, n, byrow = TRUE) + st$F %*% t(st$V)
  x <- if (identical(window, "bulk")) {
    .bulk_window(M_all, sd, points, delta)
  } else {
    seq(min(M_all) - st$left * max(sd), max(M_all) + st$right * max(sd),
        length.out = points)
  }
  dx <- x[2] - x[1]
  p <- numeric(n)
  slope <- numeric(n)
  xm <- matrix(x, n, points, byrow = TRUE)
  for (q in seq_len(Q)) {
    z <- (xm - M_all[q, ]) / sd
    b <- st$fn(z)
    f <- b$f / sd
    logS <- log(b$S)
    L <- colSums(logS)
    rest <- exp(pmin(pmax(matrix(L, n, points, byrow = TRUE) - logS,
                          -745), 0))
    p <- p + st$W[q] * rowSums(f * rest) * dx
    slope <- slope + st$W[q] * rowSums(-b$fp / sd^2 * rest) * dx
  }
  total <- sum(p)
  if (return_slopes) return(list(p = p / total, slopes = slope / total))
  p / total
}

#' Invert the general race: mean-zero mu reproducing probabilities p
#'
#' @param p positive target probabilities (normalized internally)
#' @param V,D,F,W,base,points,structure,qa,qf as in
#'   \code{\link{race_probabilities}}
#' @param n_iter maximum damped-Newton iterations
#' @param tol convergence tolerance on max |log p - log target|
#' @return mean-zero ability vector (min-wins)
#' @export
abilities_from_race <- function(p, V = NULL, D = NULL, F = NULL, W = NULL,
                                base = "normal", points = 257,
                                n_iter = 60, tol = 1e-8,
                                structure = NULL, qa = 9, qf = 15, cov = NULL) {
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    V <- fit$V; D <- fit$D; F <- fit$F; W <- fit$W
  }
  if (!is.null(structure)) {
    return(.dispatch_abilities(p, structure, base = base, points = points,
                               qa = qa, qf = qf))
  }
  target <- as.numeric(p)
  if (any(target <= 0)) stop("all target probabilities must be positive")
  target <- target / sum(target)
  logt <- log(target)
  mu <- -(logt - mean(logt)) / 2
  alpha <- if (length(target) > 2) 1.0 else 0.7
  for (it in seq_len(n_iter)) {
    ps <- race_probabilities(mu, V = V, D = D, F = F, W = W, base = base,
                             points = points, return_slopes = TRUE)
    phat <- ps$p
    sl <- ps$slopes
    resid <- log(pmax(phat, 1e-300)) - logt
    if (max(abs(resid)) < tol) break
    dlogp <- pmin(sl / pmax(phat, 1e-300), -1e-6)
    # residual-proportional step cap: a near-certain winner's residual
    # and own-slope both vanish and their noisy ratio destabilizes the
    # recentered fixed point (heavy-favorite targets 1e-4..1e-8 stalled;
    # capped they converge in a handful of iterations)
    lim <- pmin(2, 10 * abs(resid))
    mu <- mu - pmin(pmax(alpha * resid / dlogp, -lim), lim)
    mu <- mu - mean(mu)
  }
  mu
}

#' @rdname abilities_from_race
#' @export
calibrate_abilities <- function(p, V = NULL, D = NULL, F = NULL, W = NULL,
                                base = "normal", points = 257,
                                n_iter = 60, tol = 1e-8,
                                structure = NULL, qa = 9, qf = 15) {
  abilities_from_race(p, V = V, D = D, F = F, W = W, base = base,
                      points = points, n_iter = n_iter, tol = tol,
                      structure = structure, qa = qa, qf = qf)
}
