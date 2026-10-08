# Fast rectangle probabilities for factor-structured covariance.
#
# P(a <= X <= b), X ~ N(mu, VV' + diag(D)): conditional on the r-dim
# factor f, coordinates are independent, so the probability is
#   E_f [ prod_j { Phi((b_j - mu_j - v_j'f)/s_j) - Phi((a_j - ...)/s_j) } ],
# an r-dimensional smooth integral evaluated on Gauss-Hermite or Sobol
# nodes. No lattice, no simulation, milliseconds at n in the hundreds.
#
# The node rule mirrors the python reference (winning/fastmvn.py):
# Gauss-Hermite order scaled by the sharpness ratio max ||v_i||/sqrt(D_i),
# with a FAMILY escalation to scrambled Sobol (R/sobol.R) past sharpness
# 3 or rank 2 (Gauss-Hermite converges slowly on sharp integrands at any
# order; low-discrepancy sets do not). It was fixed-prime Halton: rank 7
# indexed a missing seventh prime and died in the radical inverse, and a
# rank-4 rectangle was 1.06% high at 8192 points (#143).

.gh_cache <- new.env(parent = emptyenv())

# cached: the rule depends on (r, Q) alone, and the eigensolve that
# builds it cost more than the integral it serves
.gh_nodes <- function(r, Q) {
  key <- paste(r, Q)
  hit <- .gh_cache[[key]]
  if (!is.null(hit)) return(hit)
  out <- .gh_nodes_build(r, Q)
  assign(key, out, envir = .gh_cache)
  out
}

.gh_nodes_build <- function(r, Q) {
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
    .sobol_normal(r, 2^13, seed = 0L)
  } else {
    # The order grows like sharpness SQUARED, not 8 x sharpness: a
    # cell's transition in factor space is 1/sharpness wide, and the
    # linear rule gave 0.2% relative error on a correlation-0.8 orthant
    # (#132). Rank two keeps the old 15-point floor so ordinary inputs
    # cost what they did; rank one floors at 61 (n x Q work, negligible
    # once cached). Same rule as winning/fastmvn.py.
    Q <- as.integer(min(max(ceiling(15 * sharp^2), if (r == 1) 61 else 15),
                        if (r == 1) 201 else 41))
    .gh_nodes(r, Q)
  }
}

#' Exact factor-plus-diagonal decomposition of a covariance, if one exists
#'
#' Iterated principal-factor fit for ranks 0..max_rank on the CORRELATION
#' matrix, scaled back; accepted when the reconstruction matches to tol
#' in correlation units. Returns list(V, D) or NULL.
#'
#' The search used an absolute residual floor (1e-12) and a tolerance
#' relative to the largest entry, so 1e-12 * S was rejected while S was
#' accepted (#368), and a 0.9 correlation beside an unrelated 1e16
#' variance counted as negligible (#132). In correlation units every
#' tolerance is per pair and dimensionless. Rank zero is tried first: a
#' diagonal sigma has no factors, and a rank-one fit of it moved a
#' variance into an artificial loading (#132).
factorize_covariance <- function(sigma, max_rank = 6L, tol = 1e-11,
                                 n_iter = 300L) {
  sigma <- as.matrix(sigma)
  n <- nrow(sigma)
  if (ncol(sigma) != n || any(!is.finite(sigma))) return(NULL)
  d <- diag(sigma)
  if (any(d < 0)) return(NULL)
  pos <- d > 0
  if (any(sigma[!pos, ] != 0) || any(sigma[, !pos] != 0)) return(NULL)
  m <- sum(pos)
  sd <- sqrt(d[pos])
  R <- sigma[pos, pos, drop = FALSE] / outer(sd, sd)
  back <- function(Vr, Dr) {
    V <- matrix(0, n, ncol(Vr)); V[pos, ] <- Vr * sd
    D <- numeric(n); D[pos] <- Dr * d[pos]
    list(V = V, D = D)
  }
  if (m == 0 || max(abs(R - diag(diag(R), m))) <= tol)
    return(list(V = matrix(0, n, 0), D = d))
  dR <- diag(R)
  for (r in seq_len(min(max_rank, m - 1L))) {
    D <- rep(0.5, m)
    for (it in seq_len(n_iter)) {
      e <- eigen(R - diag(D, m), symmetric = TRUE)
      idx <- order(e$values, decreasing = TRUE)[seq_len(r)]
      V <- e$vectors[, idx, drop = FALSE] *
        rep(sqrt(pmax(e$values[idx], 0)), each = m)
      D_new <- pmax(dR - rowSums(V^2), 1e-12)
      if (max(abs(D_new - D)) < 1e-12) { D <- D_new; break }
      D <- D_new
    }
    # final verification with V recomputed against the accepted D: the
    # decomposition is used only if the reconstruction is essentially
    # exact, otherwise the caller falls back to mvtnorm -- a loose fit
    # must never masquerade as the structured case.
    e <- eigen(R - diag(D, m), symmetric = TRUE)
    idx <- order(e$values, decreasing = TRUE)[seq_len(r)]
    V <- e$vectors[, idx, drop = FALSE] *
      rep(sqrt(pmax(e$values[idx], 0)), each = m)
    if (max(abs(V %*% t(V) + diag(D, m) - R)) < tol)
      return(back(V, D))
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
# Refuse a reversed rectangle {lower <= x <= upper}.
#
# A reversed coordinate makes the event EMPTY. Every port used to form the
# negative conditional cell -- pnorm(0) - pnorm(1) = -0.3413... -- and then
# clamp it to the underflow floor 1e-300, so an impossible observation came
# back as a finite probability (#235). mvtnorm::pmvnorm raises on reversed
# bounds; all four ports agree with it.
#
# lower == upper is a zero-mass slab; .drop_constants returns it.
.check_bounds <- function(lower, upper) {
  bad <- which(lower > upper)
  if (length(bad)) {
    i <- bad[1]
    stop(sprintf(paste0("lower must not exceed upper: coordinate %d has ",
                        "lower=%g > upper=%g, so the rectangle is empty. ",
                        "Check the argument order."),
                 i, lower[i], upper[i]), call. = FALSE)
  }
  invisible(TRUE)
}

# Settle the degenerate cases before any integration. lower == upper on
# any coordinate is a zero-mass slab, as in mvtnorm::pmvnorm. A
# zero-variance coordinate is the constant mean_i: outside its interval
# the rectangle is empty; inside, it leaves the integral. Near-
# deterministic (Dirac) inputs are otherwise out of scope.
.drop_constants <- function(var, mean, lower, upper) {
  if (any(lower == upper))
    return(list(done = structure(0, method = "degenerate-rectangle")))
  const <- var == 0
  if (any(const & (mean < lower | mean > upper)))
    return(list(done = structure(0, method = "outside-support")))
  keep <- !const
  if (!any(keep)) return(list(done = structure(1, method = "factor")))
  list(keep = keep, done = NULL)
}

# One coordinate: the marginal is N(mean, var) whatever the factor
# decomposition, so no quadrature (#395).
.univariate <- function(lower, upper, mean, var) {
  lm <- .log_cell_mass(matrix(upper - mean, 1), matrix(lower - mean, 1),
                       sqrt(var))
  structure(exp(lm[1, 1]), method = "factor", nodes = 1L)
}

# Per-coordinate LOG conditional cell mass, s == 0 included.
#
# In the log domain, on the side of zero where nothing cancels. The mass
# used to be pnorm(hi) - pnorm(lo): in the upper tail both round to 1, so
# P(9 < Z <= 10) = 1.13e-19 became 0 and then the 1e-300 floor, and a
# rectangle and its reflection got different probabilities (#98, #196).
# Upper-tail intervals are reflected to the lower tail and
# pnorm(log.p = TRUE) carries the deep tail, so nothing needs a floor: an
# exactly empty cell is -Inf and stays -Inf.
#
# A coordinate with zero idiosyncratic variance is DETERMINISTIC given the
# factor draw, so its cell is an INDICATOR (#206). `hi` and `lo` are
# matrices already shifted by the mean and the factor term; `s` is the
# per-coordinate sd, recycled down the columns.
.log_cell_mass <- function(hi, lo, s) {
  sdm <- matrix(rep(s, each = nrow(hi)), nrow = nrow(hi))
  if (all(s > 0) && all(lo == -Inf)) {
    # the common one-sided cell: log Phi(hi/s), stable on both sides
    return(matrix(pnorm(hi / sdm, log.p = TRUE), nrow(hi), ncol(hi)))
  }
  out <- matrix(-Inf, nrow(hi), ncol(hi))
  pos <- sdm > 0
  det <- !pos
  if (any(det))
    out[det] <- ifelse(lo[det] <= 0 & 0 <= hi[det], 0, -Inf)
  if (any(pos)) {
    b <- hi[pos] / sdm[pos]
    a <- lo[pos] / sdm[pos]
    flip <- a > 0
    a2 <- ifelse(flip, -b, a)
    b2 <- ifelse(flip, -a, b)
    lb <- pnorm(b2, log.p = TRUE)
    la <- pnorm(a2, log.p = TRUE)
    left <- suppressWarnings(lb + log1p(-exp(la - lb)))
    mid <- log1p(-(pnorm(a2) + pnorm(-b2)))
    lm <- ifelse(b2 <= 0, left, mid)
    lm[a2 >= b2] <- -Inf
    out[pos] <- lm
  }
  out
}

# Gradient ascent with backtracking on the log-integrand. The old search
# differentiated a log of cells floored at 1e-300, whose gradient is
# exactly zero wherever every cell has underflowed; the tail repair that
# #395 routes unresolved estimates to depends on finding the mode.
.ascend <- function(logint, f0, h = 1e-4, iters = 100L) {
  r <- length(f0)
  val <- logint(f0)
  for (it in seq_len(iters)) {
    g <- vapply(seq_len(r), function(k) {
      ek <- replace(rep(0, r), k, h)
      (logint(f0 + ek) - logint(f0 - ek)) / (2 * h)
    }, 0)
    if (any(!is.finite(g)) || sqrt(sum(g^2)) < 1e-8) break
    step <- 1
    moved <- FALSE
    while (step > 1e-12) {
      cand <- f0 + pmin(pmax(step * g, -1), 1)
      cv <- logint(cand)
      if (cv > val) { f0 <- cand; val <- cv; moved <- TRUE; break }
      step <- step / 2
    }
    if (!moved) break
  }
  f0
}

# One place decides what "one value per coordinate" means here, the way
# winning.shapes.as_idio does for python. A SCALAR is broadcast on
# purpose -- that is what the -Inf/Inf defaults are -- and every other
# wrong length is refused.
#
# R recycles silently whenever the short length divides n, and no
# warning is emitted, so `D = c(1, 4)` at n = 4 became c(1,4,1,4) and
# the call returned [Phi(1)Phi(1/2)]^2 = 0.3384: a perfectly plausible
# probability for a Gaussian the caller never asked about. `mean` and
# `upper` did the same (#285). The python reference validates through
# as_idio/as_loadings and julia fails on unequal lengths; only this
# port guessed.
.as_len <- function(x, n, what, finite = TRUE, nonneg = FALSE) {
  x <- as.numeric(x)
  if (length(x) == 1L) x <- rep(x, n)
  if (length(x) != n)
    stop(sprintf(paste("%s must be a scalar or one value per coordinate;",
                       "got %d for n = %d"), what, length(x), n),
         call. = FALSE)
  if (anyNA(x))
    stop(sprintf("%s has a missing entry at %d", what, which(is.na(x))[1]),
         call. = FALSE)
  if (finite && any(!is.finite(x)))
    stop(sprintf("%s has a non-finite entry at %d", what,
                 which(!is.finite(x))[1]), call. = FALSE)
  if (nonneg && any(x < 0))
    stop(sprintf("%s[%d] = %g is a negative variance", what,
                 which(x < 0)[1], x[which(x < 0)[1]]), call. = FALSE)
  x
}

pmvnorm_fast <- function(lower = -Inf, upper = Inf, mean = NULL,
                         sigma = NULL, V = NULL, D = NULL, ...) {
  if (is.null(V) || is.null(D)) {
    if (is.null(sigma)) stop("supply sigma, or V and D")
    # Normalise every per-coordinate argument ONCE, before dispatch: the
    # dense fallback used to forward the raw `mean` to mvtnorm, whose
    # checker recycles it (cbind(lower, upper, mean)), so mean = c(0, 1)
    # at n = 8 priced rep(c(0, 1), 4) in silence -- 9.6e-4 where the
    # zero-mean answer is 2.31e-2 -- while the structured path refused
    # the same call (#437).
    nn <- nrow(as.matrix(sigma))
    mean <- if (is.null(mean)) rep(0, nn) else .as_len(mean, nn, "mean")
    lower <- .as_len(lower, nn, "lower", finite = FALSE)
    upper <- .as_len(upper, nn, "upper", finite = FALSE)
    fd <- factorize_covariance(sigma)
    if (is.null(fd)) {
      sigma <- as.matrix(sigma)
      nn <- nrow(sigma)
      lo <- .as_len(lower, nn, "lower", finite = FALSE)
      up <- .as_len(upper, nn, "upper", finite = FALSE)
      mu <- .as_len(mean, nn, "mean")
      .check_bounds(lo, up)
      st <- .drop_constants(diag(sigma), mu, lo, up)
      if (!is.null(st$done)) return(st$done)
      keep <- st$keep
      if (sum(keep) == 1L)
        return(.univariate(lo[keep], up[keep], mu[keep],
                           sigma[keep, keep]))
      if (all(keep)) {
        p <- mvtnorm::pmvnorm(lower = lo, upper = up, mean = mu,
                              sigma = sigma, ...)
      } else {
        p <- mvtnorm::pmvnorm(lower = lo[keep], upper = up[keep],
                              mean = mu[keep],
                              sigma = sigma[keep, keep, drop = FALSE], ...)
      }
      attr(p, "method") <- "mvtnorm-fallback"
      return(p)
    }
    V <- fd$V; D <- fd$D
  }
  V <- as.matrix(V)
  n <- nrow(V)
  # every per-coordinate argument goes through the same contract; a
  # bound may be infinite, a variance may not
  D <- .as_len(D, n, "D", nonneg = TRUE)
  mean <- if (is.null(mean)) rep(0, n) else .as_len(mean, n, "mean")
  lower <- .as_len(lower, n, "lower", finite = FALSE)
  upper <- .as_len(upper, n, "upper", finite = FALSE)
  .check_bounds(lower, upper)
  st <- .drop_constants(rowSums(V^2) + D, mean, lower, upper)
  if (!is.null(st$done)) return(st$done)
  keep <- st$keep
  if (!all(keep)) {
    V <- V[keep, , drop = FALSE]; D <- D[keep]; mean <- mean[keep]
    lower <- lower[keep]; upper <- upper[keep]
  }
  if (length(D) == 1L)
    return(.univariate(lower, upper, mean, sum(V^2) + D))
  # a factor no coordinate loads on integrates out exactly
  V <- V[, colSums(V != 0) > 0, drop = FALSE]
  s <- sqrt(D)
  if (ncol(V) == 0L) {
    lm <- .log_cell_mass(matrix(upper - mean, 1), matrix(lower - mean, 1), s)
    return(structure(exp(sum(lm)), method = "factor", nodes = 1L))
  }
  nd <- .nodes_for(V, D)
  M <- nd$F %*% t(V)                        # (Q, n) conditional shifts
  lo <- sweep(-M, 2, lower - mean, "+")     # (Q, n): lower - mean - v'f
  hi <- sweep(-M, 2, upper - mean, "+")
  contrib <- nd$W * exp(rowSums(.log_cell_mass(hi, lo, s)))
  p <- sum(contrib)
  # trusted only if RESOLVED: on the equal-weight low-discrepancy rule
  # one node is 1/8192, so the 1e-8 trigger alone accepted a single
  # accidental hit as the probability (#395)
  ess <- if (p > 0) p^2 / max(sum(contrib^2), 1e-300) else 0
  qmc <- nrow(nd$F) >= 2^13 && all(nd$W == nd$W[1])
  if (p < 1e-8 || (qmc && ess < 100)) {
    # deep tail: the integrand concentrates in a corner of factor space
    # that centered nodes cannot see. Recenter at the mode of the
    # log-integrand and importance-reweight.
    r <- ncol(V)
    logint <- function(f) {
      z <- as.vector(V %*% f)
      sum(.log_cell_mass(matrix(upper - mean - z, nrow = 1),
                         matrix(lower - mean - z, nrow = 1), s)) -
        0.5 * sum(f^2)
    }
    f0 <- .ascend(logint, rep(0, r))
    n_nodes <- 2^13
    Fh <- .sobol_normal(r, n_nodes, seed = 1L)$F
    tau <- 1.5                              # proposal sd around the mode
    Fq <- sweep(Fh * tau, 2, f0, "+")
    logw <- -0.5 * rowSums(Fq^2) + 0.5 * rowSums(Fh^2) + r * log(tau)
    Mq <- Fq %*% t(V)
    loq <- sweep(-Mq, 2, lower - mean, "+")
    hiq <- sweep(-Mq, 2, upper - mean, "+")
    # importance identity: E_phi[cell] = mean over q-draws of
    # cell(Fq) * phi(Fq)/q(Fq), and log(phi/q) = logw above
    lt <- rowSums(.log_cell_mass(hiq, loq, s)) + logw
    m <- max(lt)
    p <- if (is.finite(m)) exp(m) * mean(exp(lt - m)) else 0
    return(structure(p, method = "factor-recentered", nodes = n_nodes))
  }
  structure(p, method = "factor", nodes = nrow(nd$F))
}
