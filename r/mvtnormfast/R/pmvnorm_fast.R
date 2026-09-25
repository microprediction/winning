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
# Classify the rectangle {lower <= x <= upper} before any quadrature.
#
# A reversed coordinate makes the event EMPTY. Every port used to form the
# negative conditional cell -- pnorm(0) - pnorm(1) = -0.3413... -- and then
# clamp it to the underflow floor 1e-300, so an impossible observation came
# back as a finite probability and, in log-likelihood code, as about -690.8
# instead of -Inf (#235). mvtnorm::pmvnorm, which this is a drop-in for,
# raises on reversed bounds and returns 0 when a coordinate has
# lower == upper; all four ports now agree with it.
#
# Returns TRUE when the rectangle is degenerate (zero mass, exactly).
.rectangle_status <- function(lower, upper) {
  bad <- which(lower > upper)
  if (length(bad)) {
    i <- bad[1]
    stop(sprintf(paste0("lower must not exceed upper: coordinate %d has ",
                        "lower=%g > upper=%g, so the rectangle is empty. ",
                        "Check the argument order."),
                 i, lower[i], upper[i]), call. = FALSE)
  }
  any(lower == upper)
}

# Per-coordinate conditional cell mass, s == 0 included.
#
# A coordinate with zero idiosyncratic variance is DETERMINISTIC given the
# factor draw: X_i = mean_i + v_i . f. Its cell is an INDICATOR, not a
# normal interval, and dividing by s = 0 gave 0/0 = NaN the moment that
# deterministic value landed exactly on an inclusive rectangle boundary --
# P(X_1 <= 0) for X_1 identically 0, which is 1, not undefined (#206).
# `hi` and `lo` are matrices already shifted by the mean and the factor
# term; `s` is the per-coordinate sd, recycled down the columns.
.cell_mass <- function(hi, lo, s) {
  sdm <- matrix(rep(s, each = nrow(hi)), nrow = nrow(hi))
  out <- matrix(0, nrow(hi), ncol(hi))
  pos <- sdm > 0
  if (any(pos))
    out[pos] <- pnorm(hi[pos] / sdm[pos]) - pnorm(lo[pos] / sdm[pos])
  det <- !pos
  if (any(det))
    out[det] <- as.numeric(lo[det] <= 0 & 0 <= hi[det])
  out
}

pmvnorm_fast <- function(lower = -Inf, upper = Inf, mean = NULL,
                         sigma = NULL, V = NULL, D = NULL, ...) {
  if (is.null(V) || is.null(D)) {
    if (is.null(sigma)) stop("supply sigma, or V and D")
    if (is.null(mean)) mean <- rep(0, nrow(as.matrix(sigma)))
    fd <- factorize_covariance(sigma)
    if (is.null(fd)) {
      nn <- nrow(as.matrix(sigma))
      if (.rectangle_status(rep_len(lower, nn), rep_len(upper, nn))) {
        p <- 0
        attr(p, "method") <- "degenerate-rectangle"
        return(p)
      }
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
  if (.rectangle_status(lower, upper)) {
    p <- 0
    attr(p, "method") <- "degenerate-rectangle"
    return(p)
  }
  s <- sqrt(D)
  # A coordinate with no idiosyncratic variance AND no loading is a
  # constant at mean_i. Outside its own interval the probability is
  # exactly 0, not the 1e-300 the cell floor would give it.
  fixed <- s == 0 & apply(V != 0, 1, function(z) !any(z))
  if (any(fixed & (mean < lower | mean > upper))) {
    p <- 0
    attr(p, "method") <- "outside-support"
    return(p)
  }
  nd <- .nodes_for(V, D)
  M <- nd$F %*% t(V)                        # (Q, n) conditional shifts
  lo <- sweep(-M, 2, lower - mean, "+")     # (Q, n): lower - mean - v'f
  hi <- sweep(-M, 2, upper - mean, "+")
  logcell <- log(pmax(.cell_mass(hi, lo, s), 1e-300))
  p <- sum(nd$W * exp(rowSums(logcell)))
  if (p < 1e-8) {
    # deep tail: the integrand concentrates in a corner of factor space
    # that centered nodes cannot see. Recenter at the Laplace point
    # (Newton on the log-integrand) and importance-reweight.
    r <- ncol(V)
    logint <- function(f) {
      z <- as.vector(V %*% f)
      sum(log(pmax(.cell_mass(matrix(upper - mean - z, nrow = 1),
                              matrix(lower - mean - z, nrow = 1), s),
                   1e-300))) - 0.5 * sum(f^2)
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
    loq <- sweep(-Mq, 2, lower - mean, "+")
    hiq <- sweep(-Mq, 2, upper - mean, "+")
    lc <- log(pmax(.cell_mass(hiq, loq, s), 1e-300))
    # importance identity: E_phi[cell] = mean over q-draws of
    # cell(Fq) * phi(Fq)/q(Fq), and log(phi/q) = logw above
    lt <- rowSums(lc) + logw
    m <- max(lt)
    p <- exp(m) * mean(exp(lt - m))
    return(structure(p, method = "factor-recentered", nodes = n_nodes))
  }
  structure(p, method = "factor", nodes = nrow(nd$F))
}
