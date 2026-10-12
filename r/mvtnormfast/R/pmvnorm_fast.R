# Fast rectangle probabilities for factor-structured covariance.
#
# P(a <= X <= b), X ~ N(mu, VV' + diag(D)): conditional on the r-dim
# factor f, coordinates are independent, so the probability is
#   E_f [ prod_j { Phi((b_j - mu_j - v_j'f)/s_j) - Phi((a_j - ...)/s_j) } ],
# an r-dimensional smooth integral evaluated on Gauss-Hermite or Sobol
# nodes. No lattice, no simulation, milliseconds at n in the hundreds.
#
# The node rule mirrors the python reference (winning/fastmvn.py):
# Gauss-Hermite placed at the mode of the integrand and scaled by its
# curvature (.adaptive_rule, #470), order scaled by the sharpness ratio
# max ||v_i||/sqrt(D_i), with a FAMILY escalation to scrambled Sobol
# (R/sobol.R) past sharpness 3 or rank 2 (Gauss-Hermite converges slowly on sharp integrands at any
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
  F <- as.matrix(grids)[keep, , drop = FALSE]
  list(F = F, W = W[keep] / sum(W[keep]), X2 = rowSums(F^2))
}

# The order grows like sharpness SQUARED, not 8 x sharpness: a cell's
# transition in factor space is 1/sharpness wide, and the linear rule gave
# 0.2% relative error on a correlation-0.8 orthant (#132). Rank two floors
# at 15 points, rank one at 61. Same rule as winning/fastmvn.py.
.gh_order <- function(r, sharp)
  as.integer(min(max(ceiling(15 * sharp^2), if (r == 1) 61 else 15),
                 if (r == 1) 201 else 41))

.nodes_for <- function(V, D) {
  r <- ncol(V)
  sharp <- max(sqrt(rowSums(V^2)) / sqrt(pmax(D, 1e-300)))
  if (sharp > 3.0 || r > 2) {
    .sobol_normal(r, 2^13, seed = 0L)
  } else {
    .gh_nodes(r, .gh_order(r, sharp))
  }
}

# V with the factor basis fixed by VV' alone (winning/fastmvn.py,
# _canonical_loadings). Any V O with O orthogonal is the same Gaussian,
# but the Sobol cloud is not rotation invariant (a sign flip of one column
# moved a rank-3 answer, #518) and the raw column count chose the rule
# (two zero columns moved a rank-one answer 2.37%, #457). Columns become
# V W, W the eigenvectors of V'V in decreasing order; directions carrying
# nothing are dropped.
.canonical_loadings <- function(V) {
  V <- V[, colSums(V != 0) > 0, drop = FALSE]
  if (ncol(V) > 1L) {
    e <- eigen(crossprod(V), symmetric = TRUE)   # decreasing
    keep <- e$values > 1e-13 * e$values[1]
    V <- V %*% e$vectors[, keep, drop = FALSE]
  }
  V
}

# Each column's sign fixed by its largest entry: the Gauss-Hermite rules
# are symmetric and need none of this; the Sobol cloud is not.
.sign_fixed <- function(V) {
  if (ncol(V) == 0L) return(V)
  big <- apply(abs(V), 2, which.max)
  sgn <- ifelse(V[cbind(big, seq_len(ncol(V)))] < 0, -1, 1)
  V * rep(sgn, each = nrow(V))
}

.eigh_small <- function(P) {
  if (nrow(P) == 1L) return(list(values = P[1, 1], vectors = matrix(1, 1, 1)))
  a <- P[1, 1]; b <- 0.5 * (P[1, 2] + P[2, 1]); c <- P[2, 2]
  if (b == 0) return(list(values = c(a, c), vectors = diag(2)))
  h <- 0.5 * (a - c)
  rad <- sqrt(h^2 + b^2)
  l1 <- 0.5 * (a + c) + rad
  v <- if (h <= 0) c(b, l1 - a) else c(l1 - c, b)
  v <- v / sqrt(sum(v^2))
  list(values = c(l1, 0.5 * (a + c) - rad),
       vectors = matrix(c(v[1], v[2], -v[2], v[1]), 2, 2))
}

# Adaptive Gauss-Hermite nodes and weights at rank <= 2, or NULL (python
# reference: winning/fastmvn.py, _adaptive_rule).
#
# The integrand's width in factor space is set by the SUM of the cells'
# curvatures, not by the sharpest row: 200 individually mild rows on a
# ring made an integrand 0.28 wide, which the fixed 15 x 15 rule at unit
# scale saw through one node, 36.7% high (#470). The rule is placed at the
# mode f0 of the log-integrand (damped Newton from the origin; the
# log-integrand is concave), along the eigenvectors U of its curvature P
# there, with widths sig: f = f0 + U diag(sig) x, and the ratio of the
# standard normal to that Gaussian folded into the weights. sig starts at
# P's eigenvalues^-1/2 and is widened, never past 1, to the width the
# log-integrand's drop three widths out implies on its flatter side (a
# soft-step cell leaves only the prior on one side). f0, U and sig follow
# the Gaussian, not the factor basis, so V O prices like V (#518).
.adaptive_rule <- function(V, D, mean, lower, upper) {
  r <- ncol(V)
  if (r > 2L) return(NULL)
  s <- sqrt(D)
  Ws <- V / sqrt(pmax(D, 1e-300))
  sharp <- sqrt(max(rowSums(Ws^2)))
  if (!(sharp <= 3.0)) return(NULL)
  n <- length(D)
  cb <- (upper - mean) / s
  one_sided <- all(lower == -Inf)
  ca <- if (one_sided) NULL else (lower - mean) / s
  ones <- rep(1, n)
  l2pi <- 0.5 * log(2 * pi)
  eye <- diag(1, r)
  evaluate <- function(f) {
    m <- as.vector(Ws %*% f)
    b <- cb - m
    if (one_sided) {
      lC <- pnorm(b, log.p = TRUE)
    } else {
      a <- ca - m
      lC <- as.vector(.log_cell_mass(matrix(b, 1), matrix(a, 1), ones))
    }
    tot <- sum(lC)
    if (!is.finite(tot)) return(NULL)
    eb <- exp(-0.5 * b * b - l2pi - lC)
    if (one_sided) {
      g1 <- eb
      g2 <- -(b + eb) * eb
    } else {
      ea <- exp(-0.5 * a * a - l2pi - lC)
      g1 <- eb - ea
      ta <- a * ea; ta[ea == 0] <- 0
      tb <- b * eb; tb[eb == 0] <- 0
      g2 <- ta - tb - g1 * g1
    }
    list(val = tot - 0.5 * sum(f * f),
         G = -as.vector(crossprod(Ws, g1)) - f,
         P = eye - crossprod(Ws * g2, Ws))
  }
  solve_small <- function(P, G) {
    if (r == 1L) return(G / P[1, 1])
    c(P[2, 2] * G[1] - P[1, 2] * G[2], P[1, 1] * G[2] - P[2, 1] * G[1]) /
      (P[1, 1] * P[2, 2] - P[1, 2] * P[2, 1])
  }
  f <- rep(0, r)
  d <- evaluate(f)
  if (is.null(d)) return(NULL)
  for (it in seq_len(50L)) {
    step <- solve_small(d$P, d$G)
    dec <- sum(d$G * step)
    if (!is.finite(dec)) return(NULL)
    # dec / 2 is the predicted rise to the mode: 1e-4 is within a
    # hundredth of a width of it
    if (dec < 1e-4) break
    t <- 1
    repeat {
      fn <- f + t * step
      dn <- evaluate(fn)
      if (!is.null(dn) && dn$val >= d$val - 1e-12) break
      t <- t / 2
      if (t < 1e-6) break
    }
    if (t < 1e-6) break
    f <- fn; d <- dn
  }
  e <- .eigh_small(d$P)
  U <- e$vectors
  sig <- 1 / sqrt(pmax(e$values, 1))
  Us <- U * rep(sig, each = r)
  probe <- rbind(t(3 * Us), t(-3 * Us)) + rep(f, each = 2L * r)  # (2r, r)
  mp <- probe %*% t(Ws)                                          # (2r, n)
  bp <- rep(cb, each = 2L * r) - mp
  lC <- if (one_sided) pnorm(bp, log.p = TRUE) else
    .log_cell_mass(bp, rep(ca, each = 2L * r) - mp, ones)
  drop <- d$val - rowSums(lC) + 0.5 * rowSums(probe^2)
  eff <- 3 * sig / sqrt(2 * pmin(drop[seq_len(r)], drop[r + seq_len(r)]))
  eff[is.na(eff) | !(eff > 0)] <- 1
  sig <- pmin(pmax(sig, eff), 1)
  gh <- .gh_nodes(r, .gh_order(r, sharp))
  X <- gh$F
  F <- (X * rep(sig, each = nrow(X))) %*% t(U) + rep(f, each = nrow(X))
  logw <- 0.5 * (gh$X2 - rowSums(F^2)) + sum(log(sig))
  list(F = F, W = gh$W * exp(logw))
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
  # the controls are checked first (#658). tol is in correlation units,
  # where no off-diagonal exceeds 1: tol = Inf, or any tol >= 1, accepted
  # every matrix as diagonal and dropped all its correlations
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) ||
      tol <= 0 || tol >= 1)
    stop("tol must be a finite number in (0, 1), a tolerance on ",
         "correlations; got ", deparse(tol))
  whole <- function(v, lo) is.numeric(v) && length(v) == 1L &&
    is.finite(v) && v == floor(v) && v >= lo
  if (!whole(max_rank, 0))
    stop("max_rank must be a whole number >= 0; got ", deparse(max_rank))
  if (!whole(n_iter, 1))
    stop("n_iter must be a whole number >= 1; got ", deparse(n_iter))
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
# erf(x / sqrt(2)) = P(|Z| <= |x|) with x's sign, without forming
# 2 pnorm(x) - 1, which cancels for small x; below 1e-8 the series' first
# term is exact to double precision (and x^2 would underflow)
.erf_r2 <- function(x) {
  ifelse(abs(x) < 1e-8, x * sqrt(2 / pi),
         sign(x) * pchisq(x * x, df = 1))
}

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
    # straddling zero: 1 - Phi(a) - Phi(-b) while the mass is the larger
    # part; a narrow central interval is (erf(b/r2) + erf(-a/r2)) / 2, two
    # same-signed small terms. The complement alone rounded
    # [-1e-17, 1e-17] to exactly 0 (#664).
    out_ <- pnorm(a2) + pnorm(-b2)
    mid <- log1p(-out_)
    nar <- which(out_ >= 0.5 & a2 < 0 & b2 > 0)   # only these pay for erf
    if (length(nar))
      mid[nar] <- log(0.5 * (.erf_r2(b2[nar]) - .erf_r2(a2[nar])))
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

# Exactly one complete covariance: sigma alone, or V and D together. Any
# other mix used to be resolved in silence -- sigma beside one of V, D
# dropped the factor half, a complete (V, D) beside sigma dropped sigma --
# so a stale argument priced a different Gaussian (#480, #89).
.check_covariance_inputs <- function(sigma, V, D) {
  have <- c(sigma = !is.null(sigma), V = !is.null(V), D = !is.null(D))
  if (identical(unname(have), c(TRUE, FALSE, FALSE)) ||
      identical(unname(have), c(FALSE, TRUE, TRUE))) return(invisible(TRUE))
  if (!any(have)) stop("supply sigma, or V and D", call. = FALSE)
  stop("supply exactly one covariance: sigma alone, or V and D together ",
       "(sigma = V V' + diag(D)); got ",
       paste(names(have)[have], collapse = " and "), call. = FALSE)
}

# A finite, square, symmetric, positive-semidefinite matrix, or an error;
# returns the symmetrised matrix (python: _as_covariance). The fallback
# dropped a zero-variance coordinate by its diagonal alone, so the
# indefinite matrix [[0, 1], [1, 1]] lost its first row and came back as
# pnorm(0) = 0.5 with method "factor" (#639).
.as_covariance <- function(sigma) {
  C <- as.matrix(sigma)
  storage.mode(C) <- "double"
  if (nrow(C) != ncol(C))
    stop(sprintf("sigma must be a square matrix; got %d x %d", nrow(C),
                 ncol(C)), call. = FALSE)
  if (any(!is.finite(C))) stop("sigma contains NA, NaN or Inf", call. = FALSE)
  if (length(C) == 0L) return(C)
  asym <- max(abs(C - t(C)))
  if (asym > 1e-8 * max(max(abs(C)), 1e-300))
    stop(sprintf(paste0("sigma is not symmetric (max asymmetry %.2e); pass ",
                        "(sigma + t(sigma))/2 if the asymmetry is numerical ",
                        "noise"), asym), call. = FALSE)
  C <- 0.5 * C + 0.5 * t(C)
  lam <- min(eigen(C, symmetric = TRUE, only.values = TRUE)$values)
  if (lam < -1e-8 * max(mean(diag(C)), 1e-300))
    stop(sprintf(paste0("sigma is not positive semidefinite (min eigenvalue ",
                        "%.2e); it is not a covariance matrix"), lam),
         call. = FALSE)
  C
}

# The number of coordinates (python: _infer_dimension). It came from
# nrow(V) alone, so a scalar loading or the (rank, n) spelling redefined an
# n-coordinate rectangle as one coordinate and then refused the length-n D
# (#556). A loading MATRIX still decides by its rows -- unless no
# length-bearing argument (D, mean, lower, upper with more than one
# entry) has that many entries and one has ncol(V): the (rank, n)
# spelling. A scalar or vector V defers to those arguments, D first. Any
# remaining disagreement is reported by .as_len under the argument's name.
.infer_dimension <- function(V, D, mean, lower, upper) {
  lens <- c(length(D), length(mean), length(lower), length(upper))
  lens <- lens[lens > 1L]
  if (!is.null(dim(V)) && length(dim(V)) == 2L) {
    if (!length(lens) || nrow(V) %in% lens || !(ncol(V) %in% lens))
      return(nrow(V))
    return(ncol(V))
  }
  if (length(lens)) return(lens[1])
  length(V)
}

# Loadings as the (n, rank) matrix (python: winning.shapes.as_loadings): a
# scalar is one common loading, a length-n vector rank one, an (n, rank)
# matrix as given, a (rank, n) matrix transposed (rank != n only: a square
# V is never transposed).
.as_loadings <- function(V, n) {
  A <- V
  if (!is.numeric(A) || any(!is.finite(A)))
    stop("V must be finite numeric loadings", call. = FALSE)
  if (is.null(dim(A))) {
    if (length(A) == 1L) return(matrix(as.numeric(A), n, 1L))
    if (length(A) == n) return(matrix(as.numeric(A), n, 1L))
  } else if (length(dim(A)) == 2L) {
    A <- matrix(as.numeric(A), nrow(A), ncol(A))
    if (nrow(A) == n) return(A)
    if (ncol(A) == n) return(t(A))
  }
  stop(sprintf(paste0("V must carry one row per coordinate: expected a ",
                      "scalar, a vector of length %d, or a (%d, rank) ",
                      "matrix; got %s"), n, n,
               paste(if (is.null(dim(A))) length(A) else dim(A),
                     collapse = " x ")), call. = FALSE)
}

pmvnorm_fast <- function(lower = -Inf, upper = Inf, mean = NULL,
                         sigma = NULL, V = NULL, D = NULL, ...) {
  .check_covariance_inputs(sigma, V, D)
  if (!is.null(sigma)) {
    sigma <- .as_covariance(sigma)
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
    .check_bounds(lower, upper)
    # unconstrained coordinates leave before the decomposition (#487)
    free <- lower == -Inf & upper == Inf
    if (all(free)) return(structure(1, method = "factor"))
    if (any(free)) {
      sigma <- as.matrix(sigma)[!free, !free, drop = FALSE]
      mean <- mean[!free]; lower <- lower[!free]; upper <- upper[!free]
    }
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
  n <- .infer_dimension(V, D, mean, lower, upper)
  V <- .as_loadings(V, n)
  # every per-coordinate argument goes through the same contract; a
  # bound may be infinite, a variance may not
  D <- .as_len(D, n, "D", nonneg = TRUE)
  mean <- if (is.null(mean)) rep(0, n) else .as_len(mean, n, "mean")
  lower <- .as_len(lower, n, "lower", finite = FALSE)
  upper <- .as_len(upper, n, "upper", finite = FALSE)
  .check_bounds(lower, upper)
  # a coordinate bounded by (-Inf, Inf) has cell 1 for every factor draw:
  # marginalised out exactly before any rule is chosen (#487)
  free <- lower == -Inf & upper == Inf
  if (all(free)) return(structure(1, method = "factor"))
  if (any(free)) {
    V <- V[!free, , drop = FALSE]; D <- D[!free]; mean <- mean[!free]
    lower <- lower[!free]; upper <- upper[!free]
  }
  st <- .drop_constants(rowSums(V^2) + D, mean, lower, upper)
  if (!is.null(st$done)) return(st$done)
  keep <- st$keep
  if (!all(keep)) {
    V <- V[keep, , drop = FALSE]; D <- D[keep]; mean <- mean[keep]
    lower <- lower[keep]; upper <- upper[keep]
  }
  if (length(D) == 1L)
    return(.univariate(lower, upper, mean, sum(V^2) + D))
  V <- .canonical_loadings(V)
  s <- sqrt(D)
  if (ncol(V) == 0L) {
    lm <- .log_cell_mass(matrix(upper - mean, 1), matrix(lower - mean, 1), s)
    return(structure(exp(sum(lm)), method = "factor", nodes = 1L))
  }
  nd <- .adaptive_rule(V, D, mean, lower, upper)
  if (is.null(nd)) {
    V <- .sign_fixed(V)
    nd <- .nodes_for(V, D)
  }
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
    V <- .sign_fixed(V)
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
