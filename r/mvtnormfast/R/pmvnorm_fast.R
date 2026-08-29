# Fast rectangle probabilities for factor-structured covariance.
#
# P(a <= X <= b), X ~ N(mu, VV' + diag(D)): conditional on the r-dim
# factor f, coordinates are independent, so the probability is
#   E_f [ prod_j { Phi((b_j - mu_j - v_j'f)/s_j) - Phi((a_j - ...)/s_j) } ],
# an r-dimensional smooth integral evaluated on Gauss-Hermite or Halton
# nodes. No lattice, no simulation, milliseconds at n in the hundreds.
#
# The node rule mirrors the winning package: Gauss-Hermite order scaled
# by the sharpness ratio max ||v_i||/sqrt(D_i), with a FAMILY escalation
# to Halton past sharpness 3 (Gauss-Hermite converges slowly on sharp
# integrands at any order; low-discrepancy sets do not).

.halton_unit <- function(r, n) {
  primes <- c(2, 3, 5, 7, 11, 13)[seq_len(r)]
  vapply(primes, function(b) {
    idx <- seq_len(n) + 20L
    h <- numeric(n); f <- 1 / b; i <- idx
    while (any(i > 0)) { h <- h + f * (i %% b); i <- i %/% b; f <- f / b }
    h
  }, numeric(n))
}

.gh_nodes <- function(r, Q) {
  # Golub-Welsch via eigen of the Jacobi matrix for probabilists' Hermite
  J <- diag(0, Q)
  off <- sqrt(seq_len(Q - 1))
  J[cbind(seq_len(Q - 1), 2:Q)] <- off
  J[cbind(2:Q, seq_len(Q - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  x <- e$values
  w <- e$vectors[1, ]^2
  grids <- do.call(expand.grid, rep(list(x), r))
  wgrid <- do.call(expand.grid, rep(list(w), r))
  W <- apply(as.matrix(wgrid), 1, prod)
  keep <- W > 1e-12 * max(W)
  list(F = as.matrix(grids)[keep, , drop = FALSE], W = W[keep] / sum(W[keep]))
}

.nodes_for <- function(V, D) {
  r <- ncol(V)
  sharp <- max(sqrt(rowSums(V^2)) / sqrt(pmax(D, 1e-300)))
  if (sharp > 3.0 || r > 2) {
    n <- 2^13
    F <- qnorm(pmin(pmax(.halton_unit(r, n), 1e-12), 1 - 1e-12))
    list(F = matrix(F, ncol = r), W = rep(1 / n, n))
  } else {
    Q <- as.integer(min(max(ceiling(8 * sharp), 15), if (r == 1) 201 else 41))
    .gh_nodes(r, Q)
  }
}

#' Exact factor-plus-diagonal decomposition of a covariance, if one exists
#'
#' Iterated principal-factor fit for ranks 1..max_rank; accepted when the
#' reconstruction VV' + diag(D) matches sigma to tol (relative to its
#' largest entry). Returns list(V, D) or NULL.
factorize_covariance <- function(sigma, max_rank = 6L, tol = 1e-11,
                                 n_iter = 300L) {
  sigma <- as.matrix(sigma)
  n <- nrow(sigma)
  scale <- max(abs(sigma))
  for (r in seq_len(min(max_rank, n - 1L))) {
    D <- rep(0.5 * mean(diag(sigma)), n)
    for (it in seq_len(n_iter)) {
      e <- eigen(sigma - diag(D, n), symmetric = TRUE)
      idx <- order(e$values, decreasing = TRUE)[seq_len(r)]
      V <- e$vectors[, idx, drop = FALSE] *
        rep(sqrt(pmax(e$values[idx], 0)), each = n)
      D_new <- pmax(diag(sigma) - rowSums(V^2), 1e-12)
      if (max(abs(D_new - D)) < 1e-12 * scale) { D <- D_new; break }
      D <- D_new
    }
    # final verification with V recomputed against the accepted D: the
    # decomposition is used only if the reconstruction is essentially
    # exact, otherwise the caller falls back to mvtnorm -- a loose fit
    # must never masquerade as the structured case.
    e <- eigen(sigma - diag(D, n), symmetric = TRUE)
    idx <- order(e$values, decreasing = TRUE)[seq_len(r)]
    V <- e$vectors[, idx, drop = FALSE] *
      rep(sqrt(pmax(e$values[idx], 0)), each = n)
    if (max(abs(V %*% t(V) + diag(D, n) - sigma)) < tol * scale)
      return(list(V = V, D = D))
  }
  NULL
}

#' Fast multivariate normal rectangle probability
#'
#' Drop-in for mvtnorm::pmvnorm() on the structured slice. Supply V and D
#' for the factor representation sigma = VV' + diag(D), or supply sigma
#' and an exact decomposition is searched for (ranks 1..6); if none
#' exists the call falls back to mvtnorm::pmvnorm() unchanged. The result
#' carries attr "method" ("factor" or "mvtnorm-fallback").
pmvnorm_fast <- function(lower = -Inf, upper = Inf, mean = NULL,
                         sigma = NULL, V = NULL, D = NULL, ...) {
  if (is.null(V) || is.null(D)) {
    if (is.null(sigma)) stop("supply sigma, or V and D")
    if (is.null(mean)) mean <- rep(0, nrow(as.matrix(sigma)))
    fd <- factorize_covariance(sigma)
    if (is.null(fd)) {
      p <- mvtnorm::pmvnorm(lower = lower, upper = upper, mean = mean,
                            sigma = sigma, ...)
      attr(p, "method") <- "mvtnorm-fallback"
      return(p)
    }
    V <- fd$V; D <- fd$D
  }
  V <- as.matrix(V)
  n <- nrow(V)
  if (is.null(mean)) mean <- rep(0, n)
  lower <- rep_len(lower, n); upper <- rep_len(upper, n)
  s <- sqrt(D)
  nd <- .nodes_for(V, D)
  M <- nd$F %*% t(V)                        # (Q, n) conditional shifts
  lo <- sweep(-M, 2, lower - mean, "+")     # (Q, n): lower - mean - v'f
  hi <- sweep(-M, 2, upper - mean, "+")
  lo <- sweep(lo, 2, s, "/"); hi <- sweep(hi, 2, s, "/")
  logcell <- log(pmax(pnorm(hi) - pnorm(lo), 1e-300))
  p <- sum(nd$W * exp(rowSums(logcell)))
  if (p < 1e-8) {
    # deep tail: the integrand concentrates in a corner of factor space
    # that centered nodes cannot see. Recenter at the Laplace point
    # (Newton on the log-integrand) and importance-reweight.
    r <- ncol(V)
    logint <- function(f) {
      z <- as.vector(V %*% f)
      sum(log(pmax(pnorm((upper - mean - z) / s)
                   - pnorm((lower - mean - z) / s), 1e-300))) -
        0.5 * sum(f^2)
    }
    f0 <- rep(0, r); h <- 1e-4
    for (it in 1:50) {
      g <- vapply(seq_len(r), function(k) {
        ek <- replace(rep(0, r), k, h)
        (logint(f0 + ek) - logint(f0 - ek)) / (2 * h)
      }, 0)
      if (sqrt(sum(g^2)) < 1e-8) break
      f0 <- f0 + pmin(pmax(0.5 * g, -1), 1)
    }
    n_nodes <- 2^13
    Fh <- qnorm(pmin(pmax(.halton_unit(r, n_nodes), 1e-12), 1 - 1e-12))
    Fh <- matrix(Fh, ncol = r)
    tau <- 1.5                              # proposal sd around the mode
    Fq <- sweep(Fh * tau, 2, f0, "+")
    logw <- -0.5 * rowSums(sweep(Fq, 2, rep(0, r))^2) +
      0.5 * rowSums(Fh^2) + r * log(tau)
    Mq <- Fq %*% t(V)
    loq <- sweep(sweep(-Mq, 2, lower - mean, "+"), 2, s, "/")
    hiq <- sweep(sweep(-Mq, 2, upper - mean, "+"), 2, s, "/")
    lc <- log(pmax(pnorm(hiq) - pnorm(loq), 1e-300))
    # importance identity: E_phi[cell] = mean over q-draws of
    # cell(Fq) * phi(Fq)/q(Fq), and log(phi/q) = logw above
    lt <- rowSums(lc) + logw
    m <- max(lt)
    p <- exp(m) * mean(exp(lt - m))
    return(structure(p, method = "factor-recentered", nodes = n_nodes))
  }
  structure(p, method = "factor", nodes = nrow(nd$F))
}
