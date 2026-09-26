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

.base_logistic <- function(z) {
  cc <- pi / sqrt(3)
  u <- pmin(pmax(cc * z, -700), 700)
  S <- 1 / (1 + exp(u))
  f <- cc * S * (1 - S)
  list(S = pmax(S, 1e-300), f = f, fp = -cc * f * (1 - 2 * S))
}

.base_laplace <- function(z) {
  b <- 1 / sqrt(2)
  f <- exp(-abs(z) / b) / (2 * b)
  S <- ifelse(z < 0, 1 - 0.5 * exp(z / b), 0.5 * exp(-z / b))
  list(S = pmax(S, 1e-300), f = f, fp = -sign(z) * f / b)
}

.BASES <- list(normal = .base_normal, gumbel = .base_gumbel,
               logistic = .base_logistic, laplace = .base_laplace)
.SPANS <- list(normal = c(8, 8), gumbel = c(22, 8),
               logistic = c(16, 16), laplace = c(18, 18))

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
  # One idiosyncratic variance per contestant, or a scalar broadcast on
  # purpose -- the contract winning.shapes.as_idio states. R recycles a
  # short vector in silence whenever its length divides n, so
  # D = c(2, 9) at n = 4 priced exactly the race D = c(2, 9, 2, 9)
  # prices: same numbers, no warning. Found by the cross-port
  # divergence scan, where python, the browser and julia all refuse it.
  if (is.null(D)) D <- rep(1, n)
  D <- as.numeric(D)
  if (length(D) == 1L) D <- rep(D, n)
  if (length(D) != n)
    stop(sprintf(paste("D must be a scalar or one idiosyncratic variance",
                       "per contestant; got %d for %d contestants"),
                 length(D), n), call. = FALSE)
  if (any(!is.finite(D)))
    stop("D has a non-finite entry", call. = FALSE)
  if (any(D < 0))
    stop(sprintf("D[%d] = %g is a negative variance", which(D < 0)[1],
                 D[which(D < 0)[1]]), call. = FALSE)
  if (is.null(V)) {
    V <- matrix(0, n, 1)
    F <- matrix(0, 1, 1)
    W <- 1
  } else {
    # a SCALAR V is the same loading for everyone -- the spelling
    # winning.shapes.as_loadings documents, which python and the browser
    # accept and R refused (cross-port divergence scan). as.matrix() on
    # a length-1 vector gives a 1x1, which is one contestant with one
    # factor, not n contestants sharing a loading.
    if (length(V) == 1L && is.null(dim(V))) V <- matrix(as.numeric(V), n, 1L)
    V <- as.matrix(V)
    # (rank, n) is the SAME race as (n, rank), which as_loadings
    # documents and abilities_from_race already did for itself a few
    # hundred lines down -- .race_setup did not, so R was inconsistent
    # with the contract AND with itself. The ambiguity is real only at
    # rank == n, where the contract wins: n rows is n contestants.
    if (nrow(V) != n && ncol(V) == n) V <- t(V)
    if (nrow(V) != n)
      stop(sprintf("V must have one row per contestant; got %d for %d",
                   nrow(V), n), call. = FALSE)
    # gauge-fix matching the python reference: a common loading column
    # shifts every conditional mean equally and cannot move an argmin,
    # so center V and make node selection and the lattice window
    # invariant under V -> V + 1 c'
    V <- sweep(V, 2, colMeans(V))
    if (is.null(F) || is.null(W)) {
      # adaptive order matching the python reference: sharp conditional
      # races (small D relative to loadings) need more factor nodes.
      # Dispatch on the pairwise-safe bound sqrt(2) max |(PV)_i|/sqrt(D_i),
      # which bounds the pairwise contrast sharpness from above
      sharp <- sqrt(2) * max(sqrt(rowSums(V^2)) / sqrt(pmax(D, 1e-300)))
      r <- ncol(V)
      # per-rank (Gauss-Hermite order cap, sharpness past which even that
      # order loses to the low-discrepancy family); see GH_RULE in the
      # python reference for the measurements behind each number
      cap <- if (r == 1) 201 else if (r == 2) 41 else if (r == 3) 31 else 15
      sharp_max <- if (r == 2) 3.75 else if (r == 3) 4.75 else 3.0
      if (r >= 2 && sharp > sharp_max) {
        # matching the python reference: past this sharpness the factor
        # integrand is a near-step and Gauss-Hermite converges slowly at
        # any order; escalate the FAMILY to a low-discrepancy rule.
        # Python uses scrambled Sobol; here dependency-free Halton.
        hw <- .halton_normal_nodes(r, 2^13)
        F <- hw$F
        W <- hw$W
      } else if (r == 1 && ceiling(8 * sharp) > 80) {
        # rank-1 extreme sharpness (matching the python reference):
        # Gauss-Hermite is the wrong family for a near-step integrand;
        # an equal-weight midpoint-quantile grid scaled with sharpness
        # replaces it (TV 0.65 -> 6e-3 at the same node count)
        Q <- as.integer(min(ceiling(8 * sharp), 4001))
        F <- matrix(qnorm((seq_len(Q) - 0.5) / Q), ncol = 1)
        W <- rep(1 / Q, Q)
      } else if (cap^r > 1e5) {
        # high-rank tensor footgun (matching python): past a 1e5-node
        # tensor budget escalate to low-discrepancy nodes
        hw <- .halton_normal_nodes(r, 2^13)
        F <- hw$F
        W <- hw$W
      } else {
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
  # Every node carries exactly the loadings' rank, and there is one
  # weight per node. F with the wrong number of ROWS was accepted and
  # priced a different quadrature outright -- 0.64616 0.12271 0.23064
  # 0.00049 where the right answer is 0.38173 0.30061 0.13687 0.18079 --
  # too few weights returned all NA, and extra weights were ignored. A
  # short F did fail, but with "non-conformable arguments", which names
  # nothing the caller passed. Same contract as the browser's
  # asFactorNodes (#290) and julia's.
  Fm <- as.matrix(F)
  rk <- ncol(V)
  if (ncol(Fm) != rk)
    stop(sprintf(paste("F must have one column per loading; got %d for",
                       "rank %d -- a short F prices a lower-rank model"),
                 ncol(Fm), rk), call. = FALSE)
  if (nrow(Fm) < 1L)
    stop("F is empty; there are no quadrature nodes", call. = FALSE)
  if (any(!is.finite(Fm)))
    stop("F has a non-finite node", call. = FALSE)
  if (length(W) != nrow(Fm))
    stop(sprintf("W must have one weight per factor node; got %d for %d nodes",
                 length(W), nrow(Fm)), call. = FALSE)
  F <- Fm

  # W and c*W describe the SAME factor law, so normalise here, as
  # python's _setup does through winning.shapes.as_weights and as
  # abilities_from_race already does for itself further down. The
  # forward divides its accumulated shares by their total and was
  # invariant either way; race_jacobian does not, so J(W) and J(10W)
  # differed by 1.473 -- the same defect as #281 in the standalone
  # javascript module, and #290 in the browser tree.
  W <- as.numeric(W)
  Wtot <- sum(W)
  if (!is.finite(Wtot) || Wtot <= 0)
    stop(sprintf("W must have a positive total; got %s", format(Wtot)),
         call. = FALSE)
  if (any(!is.finite(W)) || any(W < 0))
    stop("W must be finite and non-negative; a factor law has no negative mass",
         call. = FALSE)
  W <- W / Wtot
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
.warn_degraded_cov <- function(fit, where) {
  # matching winning.factor.races._fit_cov: the same two failure classes,
  # and the same advice, except that python can route around it here
  if (!isTRUE(fit$degraded)) return(invisible(NULL))
  warning(sprintf(
    "%s: the cov= grammar fit is degraded (rank %d of %d%s). The %s race is then a near-step the factor nodes cannot resolve, and the covariance residual checks do not see it -- expect percent-level error (4.6e-3 measured against the python reference on an exactly 3-factor correlation at n=8). The python package routes this case to scrambled-Sobol GHK and this port has none, so the two DISAGREE here by design; use winning (python) race_probabilities(cov=) if that matters.",
    where, fit$rank, fit$n,
    if (fit$bound > 0) sprintf(", idiosyncratic floor bound on %d of %d entries", fit$bound, fit$n) else "",
    "conditional"), call. = FALSE)
  invisible(NULL)
}

.jacobi_sweeps <- function(mu, forward, scale, alpha, n_iter, tol) {
  # Own-slope-preconditioned coordinate sweeps on the mean-zero quotient,
  # matching python's races._jacobi_sweeps. `forward(mu)` returns
  # list(resid, dres): the residual in whatever space the caller inverts
  # in, model minus target, and its own-slopes (negative).
  #
  # alpha_base is the Richardson value, a persistent fact about the
  # Jacobian; penalty is caution after a sweep that failed to contract, a
  # transient, restored on the next good sweep. One number for both can
  # only ratchet down (#178). A monotone iteration riding one mode is
  # summed rather than waited out (Aitken), above 1e3 tolerances only.
  alpha_base <- alpha
  penalty <- 1
  prev <- NULL
  prev_step <- NULL
  rmax <- Inf
  for (it in seq_len(n_iter)) {
    fw <- forward(mu)
    resid <- fw$resid
    dlogp <- fw$dres
    rmax <- max(abs(resid))
    rrms <- sqrt(mean(resid^2))
    if (rmax < tol) break
    if (!is.null(prev) && rmax >= prev$rmax && rrms >= prev$rrms) {
      if (penalty > 0.1) {
        penalty <- max(0.5 * penalty, 0.1)
        mu <- prev$mu; resid <- prev$resid; dlogp <- prev$dlogp
        rmax <- prev$rmax; rrms <- prev$rrms
        prev_step <- NULL
      }
    } else if (!is.null(prev) && penalty < 1) {
      penalty <- min(1, penalty / 0.75)
    }
    prev <- list(mu = mu, resid = resid, dlogp = dlogp, rmax = rmax, rrms = rrms)
    alpha <- alpha_base * penalty
    # residual-proportional step cap in the field's scale: a near-certain
    # winner's residual and own-slope both vanish and their noisy ratio
    # destabilizes the recentered fixed point
    lim <- pmin(2, 10 * abs(resid)) * scale
    step <- pmin(pmax(alpha * resid / dlogp, -lim), lim)
    step <- step - mean(step)
    extrapolated <- FALSE
    if (!is.null(prev_step)) {
      na <- sqrt(sum(prev_step^2))
      nb <- sqrt(sum(step^2))
      if (na > 0) {
        dot <- sum(step * prev_step)
        rho <- dot / (na * na)
        cosn <- if (nb > 0) dot / (na * nb) else 0
        ratio <- nb / na
        if (rho < 0) {
          lam <- 1 - (1 - rho) / alpha
          alpha_base <- min(max(2 / (2 - lam), 0.1), 1)
        } else if (cosn > 0.999 && ratio > 0.5 && ratio < 0.999 &&
                   rmax > 1e3 * tol) {
          # collinear steps decaying geometrically: sum the tail (Aitken)
          mu <- mu - step / (1 - ratio)
          prev_step <- NULL
          extrapolated <- TRUE
        }
      }
    }
    if (!extrapolated) {
      prev_step <- step
      mu <- mu - step
    }
  }
  list(mu = mu, converged = rmax < tol, resid = rmax)
}

race_probabilities <- function(mu, V = NULL, D = NULL, F = NULL, W = NULL,
                               base = "normal", points = 257,
                               return_slopes = FALSE, structure = NULL,
                               window = "bulk", delta = 1e-12,
                               qa = 9, qf = 15, nodes = NULL, cov = NULL) {
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    # the forward normal race with no slopes is the one case answerable
    # without the fit at all; everything else needs the factor form and
    # keeps the fit with its warning (matching python)
    routable <- identical(base, "normal") && !return_slopes
    if (isTRUE(fit$degraded) && routable)
      return(.ghk_race(mu, cov)$p)
    .warn_degraded_cov(fit, "race_probabilities")
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
  smin <- min(sd)
  sharp_here <- max(sqrt(rowSums(st$V^2))) / max(smin, 1e-300)
  if (sharp_here > 25 && dx > 0.5 * smin) {
    # extreme-sharpness lattice refinement (matching python): refine to
    # ~2 points per conditional sd, capped, warn when the cap binds
    need <- ceiling((x[length(x)] - x[1]) / (0.5 * smin)) + 1
    pts2 <- min(need, 8193)
    if (pts2 > points) {
      x <- seq(x[1], x[length(x)], length.out = pts2)
      dx <- x[2] - x[1]
      points <- pts2
    }
    if (need > 8193)
      warning("conditional races sharper than the lattice can resolve ",
              "even at 8193 points; results may carry percent-level error")
  }
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
  dense <- NULL
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    if (isTRUE(fit$degraded) && identical(base, "normal")) {
      dense <- as.matrix(cov)
    } else {
      .warn_degraded_cov(fit, "abilities_from_race")
    }
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
  n <- length(target)
  # the field's contrast scale (matching the python reference): median
  # idiosyncratic variance plus the mean factor variance under the nodes
  # actually represented, so (V, F) -> (V / c, c F) is invariant
  # the target and D must describe the SAME field: a length-2 p with a
  # length-4 D answered a two-runner race and returned two numbers,
  # silently, where python, julia and the browser all refuse (cross-port
  # divergence scan)
  Dn <- if (is.null(D)) rep(1, n) else as.numeric(D)
  if (length(Dn) == 1L) Dn <- rep(Dn, n)
  if (length(Dn) != n)
    stop(sprintf(paste("D must be a scalar or one variance per target",
                       "entry; got %d for %d entries"), length(Dn), n),
         call. = FALSE)
  Vn <- if (is.null(V)) matrix(0, n, 1) else as.matrix(V)
  if (nrow(Vn) != n && ncol(Vn) == n) Vn <- t(Vn)
  Vc <- sweep(Vn, 2, colMeans(Vn))
  if (!is.null(V) && !is.null(F)) {
    Fq <- as.matrix(F)
    Wq <- if (is.null(W)) rep(1 / nrow(Fq), nrow(Fq)) else as.numeric(W) / sum(W)
    Fc <- sweep(Fq, 2, colSums(Fq * Wq))
    CovF <- t(Fc) %*% (Fc * Wq)
  } else {
    CovF <- diag(ncol(Vc))
  }
  SigV <- Vc %*% CovF %*% t(Vc)
  scale <- sqrt(median(Dn) + mean(diag(SigV)))
  if (n == 2 && identical(base, "normal")) {
    # a pair is one Gaussian contrast: closed form (matching python)
    sd_d <- sqrt(max(SigV[1, 1] + SigV[2, 2] - 2 * SigV[1, 2] + Dn[1] + Dn[2],
                     1e-300))
    gap <- sd_d * qnorm(target[1])
    return(c(-0.5 * gap, 0.5 * gap))
  }
  mu <- -(logt - mean(logt)) / 2 * scale
  top2 <- if (n > 2) sum(sort(target, decreasing = TRUE)[1:2]) else 1
  alpha <- if (n == 2 || top2 > 0.8) 0.7 else 1.0
  if (!is.null(dense)) {
    # Invert the routed map itself, not the fit: GHK yields d log p / d mu
    # in its conditioning pass, so the same sweeps apply. Inverting the fit
    # while the forward returns GHK would make the two front doors describe
    # different races, which is python's #164.
    scale <- sqrt(mean(diag(dense)))
    mu <- -(logt - mean(logt)) / 2 * scale
    fwd <- function(m) {
      g <- .ghk_race(m, dense, want_slopes = TRUE)
      list(resid = g$logp - logt, dres = pmin(g$dlogp, -1e-6))
    }
    out <- .jacobi_sweeps(mu, fwd, scale, alpha, max(n_iter, 120), tol)
    if (!out$converged)
      warning(sprintf("abilities_from_race did not converge: max |log residual| %.2e (tol %.0e)", out$resid, tol), call. = FALSE)
    return(out$mu)
  }
  fwd <- function(m) {
    ps <- race_probabilities(m, V = V, D = D, F = F, W = W, base = base,
                             points = points, return_slopes = TRUE)
    phat <- pmax(ps$p, 1e-300)
    list(resid = log(phat) - logt, dres = pmin(ps$slopes / phat, -1e-6))
  }
  out <- .jacobi_sweeps(mu, fwd, scale, alpha, n_iter, tol)
  if (!out$converged)
    warning(sprintf("abilities_from_race did not converge: max |log residual| %.2e (tol %.0e)", out$resid, tol), call. = FALSE)
  out$mu
}

#' @rdname abilities_from_race
#' @export
calibrate_abilities <- function(p, V = NULL, D = NULL, F = NULL, W = NULL,
                                base = "normal", points = 257,
                                n_iter = 60, tol = 1e-8,
                                structure = NULL, qa = 9, qf = 15,
                                cov = NULL) {
  abilities_from_race(p, V = V, D = D, F = F, W = W, base = base,
                      points = points, n_iter = n_iter, tol = tol,
                      structure = structure, qa = qa, qf = qf, cov = cov)
}
