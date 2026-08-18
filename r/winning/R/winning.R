#' Pruned product Gauss-Hermite nodes for E over N(0, I_k)
#'
#' Golub-Welsch nodes of the probabilists' Hermite rule, taken as a
#' k-fold product grid and pruned of negligible weights.
#'
#' @param k factor dimension
#' @param order univariate quadrature order (default 15)
#' @param prune drop nodes with weight below prune * max weight
#' @return list with matrix F (nodes x k) and vector W of weights
#' @export
hermite_nodes <- function(k, order = 15, prune = 1e-7) {
  off <- sqrt(seq_len(order - 1))
  J <- matrix(0, order, order)
  J[cbind(seq_len(order - 1), seq_len(order - 1) + 1)] <- off
  J[cbind(seq_len(order - 1) + 1, seq_len(order - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  x1 <- e$values
  w1 <- e$vectors[1, ]^2
  grids <- do.call(expand.grid, rep(list(x1), k))
  wgrid <- do.call(expand.grid, rep(list(w1), k))
  W <- apply(as.matrix(wgrid), 1, prod)
  keep <- W > prune * max(W)
  F <- as.matrix(grids)[keep, , drop = FALSE]
  W <- W[keep]
  list(F = unname(F), W = W / sum(W))
}

#' All win probabilities of a factor Gaussian race (min wins)
#'
#' Computes p_i = P(X_i = min_j X_j) for X = mu + V f + sqrt(D) eps with
#' f standard k-variate normal and eps independent standard normal, in
#' one shared-lattice pass per factor node. For argmax races (utilities),
#' negate mu.
#'
#' @param mu vector of locations (length N)
#' @param V N x k matrix of factor loadings
#' @param D vector of idiosyncratic variances
#' @param nodes optional list(F, W) of factor nodes; default
#'   hermite_nodes(ncol(V))
#' @param points lattice size L (default 501)
#' @return vector of win probabilities summing to one
#' @export
win_probabilities_factor <- function(mu, V, D, nodes = NULL, points = 501) {
  V <- as.matrix(V)
  N <- length(mu)
  if (is.null(nodes)) nodes <- hermite_nodes(ncol(V))
  F <- nodes$F; W <- nodes$W
  sd <- sqrt(D)
  M <- matrix(mu, nrow(F), N, byrow = TRUE) + F %*% t(V)
  pad <- 8 * max(sd)
  lo <- min(M) - pad; hi <- max(M) + pad
  x <- seq(lo, hi, length.out = points)
  dx <- x[2] - x[1]
  p <- numeric(N)
  for (q in seq_len(nrow(F))) {
    z <- (matrix(x, N, points, byrow = TRUE) - M[q, ]) / sd
    logS <- pnorm(z, lower.tail = FALSE, log.p = TRUE)
    f <- dnorm(z) / sd
    field <- colSums(logS)
    rest <- exp(pmin(pmax(sweep(-logS, 2, field, "+"), -745), 0))
    p <- p + W[q] * rowSums(f * rest) * dx
  }
  p / sum(p)
}

#' Invert observed shares to abilities under a factor Gaussian race
#'
#' Damped coordinate Newton against the frozen shared field, with
#' analytic per-coordinate slopes and an independent-race warm start.
#' Returns the mean-zero ability vector whose model shares match p.
#'
#' @param p vector of positive target shares (normalized internally)
#' @param V N x k matrix of factor loadings
#' @param D vector of idiosyncratic variances
#' @param nodes optional list(F, W) of factor nodes
#' @param n_iter maximum Newton iterations (default 50)
#' @param tol convergence tolerance on max log-share residual over
#'   identified alternatives (default 1e-6)
#' @param points lattice size L (default 501)
#' @return mean-zero ability vector (min-wins convention)
#' @export
abilities_from_probabilities_factor <- function(p, V, D, nodes = NULL,
                                                n_iter = 50, tol = 1e-6,
                                                points = 501) {
  if (any(p <= 0)) stop("all target probabilities must be positive")
  p <- p / sum(p)
  logp <- log(p)
  V <- as.matrix(V)
  N <- length(p)
  sd <- sqrt(D)
  if (is.null(nodes)) nodes <- hermite_nodes(ncol(V))
  F <- nodes$F; W <- nodes$W
  floor_ <- max(1e-9, 1e-4 / N)
  ident <- p > floor_
  if (any(V != 0)) {
    sd_tot2 <- D + rowSums(V^2)
    mu <- abilities_from_probabilities_factor(
      p, matrix(0, N, 1), sd_tot2,
      nodes = list(F = matrix(0, 1, 1), W = 1),
      n_iter = n_iter, tol = tol, points = points)
  } else {
    mu <- (logp - mean(logp)) / 2
  }
  step_cap <- sqrt(D + rowSums(V^2))
  prev_res <- Inf
  damp <- 1
  for (it in seq_len(n_iter)) {
    M <- matrix(mu, nrow(F), N, byrow = TRUE) + F %*% t(V)
    pad <- 8 * max(sd)
    x <- seq(min(M) - pad, max(M) + pad, length.out = points)
    dx <- x[2] - x[1]
    phat <- numeric(N)
    slope <- numeric(N)
    for (q in seq_len(nrow(F))) {
      z <- (matrix(x, N, points, byrow = TRUE) - M[q, ]) / sd
      logS <- pnorm(z, lower.tail = FALSE, log.p = TRUE)
      f <- dnorm(z) / sd
      field <- colSums(logS)
      rest <- exp(pmin(pmax(sweep(-logS, 2, field, "+"), -745), 0))
      phat <- phat + W[q] * rowSums(f * rest) * dx
      slope <- slope + W[q] * rowSums(z * f / sd * rest) * dx
    }
    phat <- pmax(phat / sum(phat), 1e-300)
    resid <- log(phat) - logp
    res <- if (any(ident)) max(abs(resid[ident])) else max(abs(resid))
    if (res < tol) break
    if (res > prev_res * 1.2) damp <- max(0.25, damp * 0.5)
    prev_res <- res
    dlogp <- pmin(slope / phat, -1e-3 / (sd + 1e-9))
    delta <- pmin(pmax(damp * resid / dlogp, -step_cap), step_cap)
    mu <- mu - delta
    mu <- mu - mean(mu)
  }
  mu
}
