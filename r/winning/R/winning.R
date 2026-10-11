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
# An integer count at a node-rule door, or a refusal naming it. Both
# counts reach the product grid as a repeat count, where a nonsense
# value does not raise: it builds a WELL-FORMED rule for a different
# problem, or raises from deep inside expand.grid (#68).
.as_count <- function(x, name, minimum) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x != floor(x))
    stop(sprintf("hermite_nodes needs an integer %s; got %s", name,
                 paste(format(x), collapse = " ")), call. = FALSE)
  x <- as.integer(x)
  if (x < minimum)
    stop(sprintf("hermite_nodes needs %s >= %d; got %d", name, minimum, x),
         call. = FALSE)
  x
}

#' @details k = 0 is the EMPTY PRODUCT: one node of weight 1 with no
#'   columns, so a zero-rank loading matrix integrates over the
#'   zero-dimensional factor space and prices the independent race
#'   through this same path. It is the value of the integral, not a
#'   degenerate case to reject.
hermite_nodes <- function(k, order = 15, prune = 1e-7) {
  k <- .as_count(k, "k", 0L)
  order <- .as_count(order, "order", 1L)
  if (k == 0L) return(list(F = matrix(numeric(0), 1L, 0L), W = 1))
  h <- .hermite1(order)
  x1 <- h$nodes
  w1 <- h$weights
  if (k == 1) return(list(F = matrix(x1, ncol = 1), W = w1))
  # match the reference ordering (first coordinate slowest), and prune the
  # product grid THEN renormalize, as the reference does: pruning drops
  # ~1e-7 of the mass and a direct weighted mixture consumes W as-is. This
  # comment used to claim the reference did not renormalize, which stopped
  # being true without the port following; the weights summed to
  # 1 - 2e-9 here against exactly 1 there, found by the surface audit.
  idx <- as.matrix(do.call(expand.grid, rep(list(seq_along(x1)), k)))[, k:1,
                                                                     drop = FALSE]
  F <- matrix(x1[idx], nrow(idx), k)
  W <- apply(matrix(w1[idx], nrow(idx), k), 1, prod)
  keep <- W > prune * max(W)
  Wk <- W[keep]
  list(F = unname(F[keep, , drop = FALSE]), W = Wk / sum(Wk))
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
# (F, W) validated as python's shapes.as_factor_law: finite, non-negative,
# positive total, normalised by the largest entry BEFORE summing (#263: a
# [2, -1] rule priced shares outside [0, 1], [1e308, 1e308] overflowed),
# and zero-weight nodes dropped -- they are no-ops for the integral but
# widened the global window (#416)
.as_factor_law <- function(F, W) {
  F <- as.matrix(F); W <- as.numeric(W)
  if (length(W) != nrow(F))
    stop(sprintf("W must have one weight per factor node; got %d for %d",
                 length(W), nrow(F)), call. = FALSE)
  if (any(!is.finite(W)) || any(W < 0))
    stop("W must be finite and non-negative; a factor law has no negative mass",
         call. = FALSE)
  if (!length(W) || max(W) <= 0)
    stop("W must have a positive total", call. = FALSE)
  W <- W / max(W)
  keep <- W > 0
  list(F = F[keep, , drop = FALSE], W = W[keep] / sum(W[keep]))
}

win_probabilities_factor <- function(mu, V, D, nodes = NULL, points = 501) {
  points <- .as_points(points)
  V <- as.matrix(V)
  N <- length(mu)
  if (nrow(V) != N && ncol(V) == N) V <- t(V)
  # gauge-fix as python: a common loading column cannot move an argmin
  V <- sweep(V, 2, colMeans(V))
  if (is.null(nodes)) nodes <- hermite_nodes(ncol(V))
  law <- .as_factor_law(nodes$F, nodes$W)
  F <- law$F; W <- law$W
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
  if (!is.finite(sum(p)) || sum(p) <= 0)
    stop("factor race integration failed: every density fell between ",
         "lattice points; raise points", call. = FALSE)
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
  points <- .as_points(points)
  n_iter <- .as_iter_budget(n_iter)
  tol <- .as_tolerance(tol)
  if (any(!is.finite(p))) stop("target probabilities must be finite")
  if (any(p <= 0)) stop("all target probabilities must be positive")
  p <- p / max(p)
  p <- p / sum(p)
  logp <- log(p)
  V <- as.matrix(V)
  N <- length(p)
  if (nrow(V) != N && ncol(V) == N) V <- t(V)
  # the forward's gauge, BEFORE the warm start and step caps (#70)
  V <- sweep(V, 2, colMeans(V))
  sd <- sqrt(D)
  if (is.null(nodes)) nodes <- hermite_nodes(ncol(V))
  law <- .as_factor_law(nodes$F, nodes$W)
  F <- law$F; W <- law$W
  floor_ <- max(1e-9, 1e-4 / N)
  ident <- p > floor_
  if (any(V != 0)) {
    sd_tot2 <- D + rowSums(V^2)
    # a warm start: its own exhaustion is not the caller's (#538)
    mu <- suppressWarnings(abilities_from_probabilities_factor(
      p, matrix(0, N, 1), sd_tot2,
      nodes = list(F = matrix(0, 1, 1), W = 1),
      n_iter = n_iter, tol = tol, points = points))
  } else {
    # min-wins and in the field's units (#100)
    mu <- -(logp - mean(logp)) / 2 * sqrt(stats::median(D))
  }
  step_cap <- sqrt(D + rowSums(V^2))
  prev_res <- Inf
  # python's dominant-pair gate: the undamped step two-cycles there
  top2 <- if (N > 2) sum(sort(p, decreasing = TRUE)[1:2]) else 1
  damp <- if (top2 > 0.8) 0.7 else 1
  converged <- FALSE
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
    if (res < tol) { converged <- TRUE; break }
    if (res > prev_res * 1.2) damp <- max(0.25, damp * 0.5)
    prev_res <- res
    dlogp <- pmin(slope / phat, -1e-3 / (sd + 1e-9))
    delta <- pmin(pmax(damp * resid / dlogp, -step_cap), step_cap)
    mu <- mu - delta
    mu <- mu - mean(mu)
  }
  if (!converged) {
    # the budget ran out after a step the loop never priced: check the
    # iterate actually returned, and say so -- exhaustion used to return
    # silently (#538)
    ph <- win_probabilities_factor(mu, V, D, nodes = list(F = F, W = W),
                                   points = points)
    r <- log(pmax(ph, 1e-300)) - logp
    res <- if (any(ident)) max(abs(r[ident])) else max(abs(r))
    if (!(res < tol))
      warning(sprintf(paste("abilities_from_probabilities_factor did not",
                            "converge: max |log residual| %.2e after %d",
                            "iterations (tol %.0e)"), res, n_iter, tol),
              call. = FALSE)
  }
  mu
}
